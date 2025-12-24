import tensorflow as tf
from loss import Loss
import numpy as np

SR = 48000
COND_DIM = 3 + 1  # screech, acid, pluck + pitch
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)  # 62400

# ★改善1: 損失関数の重みを音高制御向けに調整
recon_weight = 3.0  # 波形再構成を強化
STFT_weight = 10.0  # スペクトルを最重要に
mel_weight = 8.0  # メルスペクトルも強化
diff_weight = 2.0  # 位相情報の保持
kl_weight = 0.0001  # KLは極小から開始

# 元の256倍圧縮を維持（こちらの方が安定していた）
channels = [
    (64, 5, 4),
    (128, 5, 4),
    (256, 5, 4),
    (512, 3, 4),
]

LATENT_STEPS = TIME_LENGTH // (4 * 4 * 4 * 4)
if TIME_LENGTH % (4 * 4 * 4 * 4) != 0:
    LATENT_STEPS += 1


class PitchEmbedding(tf.keras.layers.Layer):
    """
    ★改善2: 音高を埋め込み層で学習
    連続値のまま使うより、離散的な音高概念を学習できる
    """

    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        # MIDI 36-71 = 36音階
        self.embedding = tf.keras.layers.Embedding(36, embed_dim)

    def call(self, pitch_normalized):
        # 正規化された pitch [0, 1] を MIDI番号に戻す
        pitch_midi = pitch_normalized * 35.0 + 36.0
        pitch_idx = tf.cast(tf.round(pitch_midi) - 36, tf.int32)
        pitch_idx = tf.clip_by_value(pitch_idx, 0, 35)
        return self.embedding(pitch_idx)


class EnhancedFiLM(tf.keras.layers.Layer):
    """
    ★改善3: FiLMを強化して条件付けを明示的に
    """

    def __init__(self, channels):
        super().__init__()
        # 中間層を追加してより複雑な変換を学習
        self.cond_transform = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(channels, activation="relu"),
                tf.keras.layers.Dense(channels * 2),  # gamma, beta
            ]
        )

    def call(self, x, cond):
        # x: (B, T, C)
        cond_out = self.cond_transform(cond)  # (B, C*2)
        gamma, beta = tf.split(cond_out, 2, axis=-1)

        gamma = gamma[:, None, :]  # (B, 1, C)
        beta = beta[:, None, :]

        return x * (1.0 + gamma) + beta


def build_encoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))
    cond = tf.keras.Input(shape=(cond_dim,))

    # ★改善4: 音高埋め込みを使用
    pitch_normalized = cond[:, 0:1]
    timbre_cond = cond[:, 1:]

    pitch_embed_layer = PitchEmbedding(embed_dim=32)
    pitch_embed = pitch_embed_layer(pitch_normalized)
    pitch_embed = tf.squeeze(pitch_embed, axis=1)

    # 音高埋め込みと音色条件を結合
    full_cond = tf.concat([pitch_embed, timbre_cond], axis=-1)

    x = x_in

    for i, (ch, k, s) in enumerate(channels):
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = EnhancedFiLM(ch)(x, full_cond)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # ★改善5: 中間層にもスキップコネクションで情報保持
        if i > 0:
            x = tf.keras.layers.Dropout(0.1)(x)  # 過学習防止

    z_mean = tf.keras.layers.Conv1D(latent_dim, 3, padding="same")(x)

    # ★改善6: logvarの初期値と範囲を制限
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
    )(x)
    z_logvar = tf.keras.layers.Lambda(
        lambda t: tf.clip_by_value(t, -10.0, 2.0)
    )(z_logvar)

    return tf.keras.Model([x_in, cond], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # 音高埋め込み（エンコーダと同じ処理）
    pitch_normalized = cond[:, 0:1]
    timbre_cond = cond[:, 1:]

    pitch_embed_layer = PitchEmbedding(embed_dim=32)
    pitch_embed = pitch_embed_layer(pitch_normalized)
    pitch_embed = tf.squeeze(pitch_embed, axis=1)

    full_cond = tf.concat([pitch_embed, timbre_cond], axis=-1)

    x = z_in

    for i, (ch, k, s) in enumerate(reversed(channels)):
        x = tf.keras.layers.UpSampling1D(s)(x)
        x = tf.keras.layers.Conv1D(ch, k, padding="same")(x)
        x = EnhancedFiLM(ch)(x, full_cond)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # ★改善7: 最終層のカーネルサイズを調整
    # より滑らかな波形を生成
    out = tf.keras.layers.Conv1D(1, 11, padding="same", activation="tanh")(x)
    out = out[:, :TIME_LENGTH, :]

    return tf.keras.Model([z_in, cond], out, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = build_encoder(cond_dim, latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
        self.kl_anneal_step = self.add_weight(
            name="kl_weight", shape=(), initializer="zeros", trainable=False
        )

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
            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))

            kl = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★改善8: より緩やかなKLアニーリング
            # 20000ステップかけて徐々に上げる
            kl_anneal = tf.minimum(
                1.0, tf.cast(self.kl_anneal_step, tf.float32) / 8700.0
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
        grads, global_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.kl_anneal_step.assign_add(1)

        return {
            "loss": loss,
            "recon": recon,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "diff_loss": diff_loss,
            "kl": kl,
            "kl_weight": current_kl_weight,
            "grad_norm": global_norm,
        }
