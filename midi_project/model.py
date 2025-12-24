import tensorflow as tf
from loss import Loss

SR = 48000
COND_DIM = 3 + 1  # screech, acid, pluck + pitch
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)  # 62400

recon_weight = 10.0  # 波形再構成を強化
STFT_weight = 5.0
mel_weight = 5.0
diff_weight = 1.0
kl_weight = 0.0001  # 初期値を極小に（アニーリング推奨）

# ★改善1: より緩やかな圧縮 (2*2*2*2 = 16倍)
# 潜在変数の時間解像度: 48000/16 = 3000Hz → 音高情報を保持可能
channels = [
    (64, 5, 2),  # stride 2
    (128, 5, 2),
    (256, 5, 2),
    (512, 3, 2),
]

# エンコーダ出力のステップ数を計算
LATENT_STEPS = TIME_LENGTH // (2 * 2 * 2 * 2)  # 3900


class FiLM(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.gamma = tf.keras.layers.Dense(channels, kernel_initializer="zeros")
        self.beta = tf.keras.layers.Dense(channels, kernel_initializer="zeros")

    def call(self, x, cond):
        g = self.gamma(cond)[:, None, :]
        b = self.beta(cond)[:, None, :]
        return x * (1.0 + g) + b


def build_encoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))
    cond = tf.keras.Input(shape=(cond_dim,))

    x = x_in

    for ch, k, s in channels:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = FiLM(ch)(x, cond)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # ★改善2: 平均と分散を分離して学習
    z_mean = tf.keras.layers.Conv1D(latent_dim, 3, padding="same")(x)
    # logvarに制約を加えて数値安定性を確保
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-1.0),
    )(x)
    z_logvar = tf.clip_by_value(z_logvar, -10.0, 2.0)

    return tf.keras.Model([x_in, cond], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    x = z_in

    for ch, k, s in reversed(channels):
        x = tf.keras.layers.UpSampling1D(s)(x)
        x = tf.keras.layers.Conv1D(ch, k, padding="same")(x)
        x = FiLM(ch)(x, cond)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # ★改善3: より大きなカーネルで滑らかな波形生成
    out = tf.keras.layers.Conv1D(1, 15, padding="same", activation="tanh")(x)
    out = out[:, :TIME_LENGTH, :]

    return tf.keras.Model([z_in, cond], out, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = build_encoder(cond_dim, latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
        # ★改善4: KLアニーリング用のトラッカー
        self.kl_anneal_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_logvar = self.encoder([x, cond])
        z = sample_z(z_mean, z_logvar)
        x_hat = self.decoder([z, cond])
        return x_hat, z_mean, z_logvar

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder([x, cond])
            z = sample_z(z_mean, z_logvar)
            x_hat = self.decoder([z, cond])
            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            # --- Loss計算 ---
            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))  # MSEに変更
            kl = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★改善5: KLアニーリング
            # 最初の10000ステップで0から徐々に増やす
            kl_anneal = tf.minimum(
                1.0, tf.cast(self.kl_anneal_step, tf.float32) / 10000.0
            )
            current_kl_weight = kl_weight * kl_anneal

            loss = (
                recon * recon_weight
                + kl * current_kl_weight
                + stft_loss * STFT_weight
                + mel_loss * mel_weight
                + diff_loss * diff_weight
            )

        grads = tape.gradient(loss, self.trainable_variables)
        # ★改善6: 勾配クリッピングで学習安定化
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.kl_anneal_step.assign_add(1)

        return {
            "loss": loss,
            "recon": recon,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "kl": kl,
            "kl_weight": current_kl_weight,
        }
