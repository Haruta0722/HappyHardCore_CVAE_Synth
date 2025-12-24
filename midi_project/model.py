import tensorflow as tf
from loss import Loss
import numpy as np

SR = 48000
COND_DIM = 3 + 1
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)

# ★改善1: 圧縮率を大幅に下げる（64倍 → 750Hz）
# これで440Hzの音も1周期に1.7ステップ確保できる
channels = [
    (64, 5, 2),  # stride 2
    (128, 5, 2),  # stride 2
    (256, 5, 2),  # stride 2
    (512, 3, 2),  # stride 2
]  # 合計: 2^4 = 16倍圧縮

LATENT_STEPS = TIME_LENGTH // 16  # 3900


# ★改善2: 明示的な周波数情報を注入
def generate_frequency_features(pitch_normalized, length, sr=SR):
    """
    音高から正弦波特徴を生成
    デコーダーがこれをヒントに使える
    """
    # MIDI番号に戻す
    midi = pitch_normalized * 35.0 + 36.0
    freq = 440.0 * tf.pow(2.0, (midi - 69.0) / 12.0)

    # 時間軸
    t = tf.range(length, dtype=tf.float32) / sr

    # 基本周波数 + 数個の倍音の位相情報
    phase = 2.0 * np.pi * freq[:, None] * t

    sin_feat = tf.sin(phase)  # (batch, time)
    cos_feat = tf.cos(phase)

    # 2倍音、3倍音も追加
    sin2 = tf.sin(2 * phase)
    sin3 = tf.sin(3 * phase)

    return tf.stack([sin_feat, cos_feat, sin2, sin3], axis=-1)  # (B, T, 4)


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

    z_mean = tf.keras.layers.Conv1D(latent_dim, 3, padding="same")(x)
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(
            -3.0
        ),  # より小さい初期分散
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

    # ★改善3: 周波数特徴を追加入力
    freq_feat_in = tf.keras.Input(shape=(TIME_LENGTH, 4))

    x = z_in

    for i, (ch, k, s) in enumerate(reversed(channels)):
        x = tf.keras.layers.UpSampling1D(s)(x)
        x = tf.keras.layers.Conv1D(ch, k, padding="same")(x)
        x = FiLM(ch)(x, cond)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

    # 最終層の直前で周波数特徴を結合
    x = tf.keras.layers.Lambda(lambda t: t[:, :TIME_LENGTH, :])(x)
    freq_feat_processed = tf.keras.layers.Conv1D(64, 3, padding="same")(
        freq_feat_in
    )
    x = tf.keras.layers.Concatenate(axis=-1)([x, freq_feat_processed])

    # 最終出力
    out = tf.keras.layers.Conv1D(1, 15, padding="same", activation="tanh")(x)

    return tf.keras.Model([z_in, cond, freq_feat_in], out, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = build_encoder(cond_dim, latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)

        # ★改善4: 段階的学習のためのフェーズ管理
        self.training_phase = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_logvar = self.encoder([x, cond])
        z = sample_z(z_mean, z_logvar)

        # 周波数特徴を生成
        pitch = cond[:, 0]
        freq_feat = generate_frequency_features(pitch, TIME_LENGTH)

        x_hat = self.decoder([z, cond, freq_feat])
        return x_hat, z_mean, z_logvar

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder([x, cond])
            z = sample_z(z_mean, z_logvar)

            # 周波数特徴
            pitch = cond[:, 0]
            freq_feat = generate_frequency_features(pitch, TIME_LENGTH)

            x_hat = self.decoder([z, cond, freq_feat])
            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            # 損失計算
            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))

            kl = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )

            # ★改善5: Free Bits（各次元で最低限の情報量を保証）
            kl_per_dim = -0.5 * (
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            kl_free_bits = tf.reduce_mean(tf.maximum(kl_per_dim, 0.5))

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★改善6: 段階的学習
            step = tf.cast(self.optimizer.iterations, tf.float32)

            # Phase 1 (0-1000 steps): 再構成のみ
            # Phase 2 (1000-4000): KLを徐々に追加
            # Phase 3 (4000+): 全損失

            kl_weight_schedule = tf.cond(
                step < 1000.0,
                lambda: 0.0,
                lambda: tf.minimum(0.005, (step - 1000.0) / 4000 * 0.005),
            )

            loss = (
                recon * 5.0  # 再構成を最重視
                + stft_loss * 10.0
                + mel_loss * 8.0
                + diff_loss * 2.0
                + kl_free_bits * kl_weight_schedule  # Free Bitsを使用
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # ★改善7: デバッグ情報を豊富に
        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))

        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "diff": diff_loss,
            "kl": kl,
            "kl_fb": kl_free_bits,  # Free Bits版
            "kl_w": kl_weight_schedule,
            "z_std": z_std,  # 潜在変数の活用度
            "grad": grad_norm,
        }
