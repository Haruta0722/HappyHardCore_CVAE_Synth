import tensorflow as tf
from loss import Loss
import numpy as np

SR = 48000
COND_DIM = 3 + 1
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)

NUM_HARMONICS = 32  # 倍音の数を増やす

# 損失関数の重み
recon_weight = 5.0
STFT_weight = 15.0
mel_weight = 10.0
kl_weight = 0.0005


def generate_harmonic_wave(fundamental_freq, amplitudes, phases, length, sr=SR):
    """
    倍音加算合成

    Args:
        fundamental_freq: (B,) - 基本周波数 [Hz]
        amplitudes: (B, num_harmonics) - 各倍音の振幅
        phases: (B, num_harmonics) - 各倍音の初期位相 [rad]
        length: int - 生成する波形の長さ
        sr: int - サンプリングレート

    Returns:
        wave: (B, length) - 合成された波形
    """
    batch_size = tf.shape(fundamental_freq)[0]
    num_harmonics = tf.shape(amplitudes)[1]

    # 時間軸
    t = tf.range(length, dtype=tf.float32) / float(sr)  # (length,)
    t = t[None, :, None]  # (1, length, 1)

    # 基本周波数を展開
    f0 = fundamental_freq[:, None, None]  # (B, 1, 1)

    # 倍音番号
    harmonic_nums = tf.range(1.0, float(num_harmonics) + 1.0, dtype=tf.float32)
    harmonic_nums = harmonic_nums[None, None, :]  # (1, 1, num_harmonics)

    # 各倍音の角周波数
    omega = 2.0 * np.pi * f0 * harmonic_nums  # (B, 1, num_harmonics)

    # 振幅と位相を展開
    amps = amplitudes[:, None, :]  # (B, 1, num_harmonics)
    phas = phases[:, None, :]  # (B, 1, num_harmonics)

    # 各倍音の波形
    harmonics = amps * tf.sin(omega * t + phas)  # (B, length, num_harmonics)

    # 加算合成
    wave = tf.reduce_sum(harmonics, axis=-1)  # (B, length)

    return wave


class EnvelopeNet(tf.keras.layers.Layer):
    """
    時間変化するエンベロープを生成
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        # 時間方向に畳み込んでエンベロープを生成
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    1, 5, padding="same", activation="sigmoid"
                ),
            ]
        )

    def call(self, z):
        # z: (B, latent_steps, latent_dim)
        envelope = self.net(z)  # (B, latent_steps, 1)

        # アップサンプリングして目標長に
        envelope = tf.image.resize(
            envelope, [self.output_length, 1], method="bilinear"
        )

        return tf.squeeze(envelope, axis=-1)  # (B, output_length)


class HarmonicAmplitudeNet(tf.keras.layers.Layer):
    """
    時間変化する倍音振幅を生成
    """

    def __init__(self, num_harmonics=NUM_HARMONICS, output_length=TIME_LENGTH):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.output_length = output_length

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    num_harmonics, 5, padding="same", activation="sigmoid"
                ),
            ]
        )

    def call(self, z, cond):
        # z: (B, latent_steps, latent_dim)
        # cond: (B, cond_dim)

        # 条件を時間方向にブロードキャスト
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])

        # 結合
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # 倍音振幅を予測
        amps = self.net(z_cond)  # (B, latent_steps, num_harmonics)

        # アップサンプリング
        amps = tf.image.resize(
            amps, [self.output_length, self.num_harmonics], method="bilinear"
        )

        return amps  # (B, output_length, num_harmonics)


class NoiseGenerator(tf.keras.layers.Layer):
    """
    条件付きノイズ生成器
    """

    def __init__(self):
        super().__init__()

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(1, 5, padding="same", activation="tanh"),
            ]
        )

    def call(self, z, cond):
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])

        z_cond = tf.concat([z, cond_broadcast], axis=-1)
        noise = self.net(z_cond)

        # アップサンプリング
        noise = tf.image.resize(noise, [TIME_LENGTH, 1], method="bilinear")

        return tf.squeeze(noise, axis=-1)  # (B, TIME_LENGTH)


# シンプルなエンコーダー
channels = [
    (64, 5, 2),
    (128, 5, 2),
    (256, 5, 2),
    (512, 3, 2),
]

LATENT_STEPS = TIME_LENGTH // 16


def build_encoder(latent_dim=LATENT_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))

    x = x_in

    for ch, k, s in channels:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    z_mean = tf.keras.layers.Conv1D(latent_dim, 3, padding="same")(x)
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-3.0),
    )(x)
    z_logvar = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -10.0, 2.0)
    )(z_logvar)

    return tf.keras.Model([x_in], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # 倍音振幅生成器
    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    harmonic_amps_time = harmonic_amp_net(
        z_in, cond
    )  # (B, TIME_LENGTH, num_harmonics)

    # エンベロープ生成器
    envelope_net = EnvelopeNet()
    envelope = envelope_net(z_in)  # (B, TIME_LENGTH)

    # ノイズ生成器
    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)  # (B, TIME_LENGTH)

    # 基本周波数を条件から取得
    pitch = cond[:, 0]
    pitch_midi = pitch * 35.0 + 36.0
    fundamental_freq = 440.0 * tf.pow(2.0, (pitch_midi - 69.0) / 12.0)

    # ★重要: 倍音合成は時間ステップごとに振幅を変化させる
    # 簡略化のため、平均振幅で生成（本来は時間変化を反映すべき）
    avg_harmonic_amps = tf.reduce_mean(
        harmonic_amps_time, axis=1
    )  # (B, num_harmonics)

    # 初期位相（学習させることも可能だが、ここではゼロ）
    phases = tf.zeros_like(avg_harmonic_amps)

    # 倍音合成
    harmonic_wave = generate_harmonic_wave(
        fundamental_freq, avg_harmonic_amps, phases, TIME_LENGTH
    )

    # ★最終合成: 倍音 × エンベロープ + ノイズ
    # 音色によって倍音とノイズの比率を変える
    timbre = cond[:, 1:]  # (B, 3) - [screech, acid, pluck]

    # screech: 倍音強め
    # acid: バランス
    # pluck: 倍音強め、ノイズ少なめ
    harmonic_ratio = 0.9  # デフォルト
    noise_ratio = 0.1

    output = harmonic_wave * envelope * harmonic_ratio + noise * noise_ratio

    # 正規化
    output = tf.tanh(output)  # [-1, 1]に制限
    output = output[:, :, None]  # (B, TIME_LENGTH, 1)

    return tf.keras.Model([z_in, cond], output, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)

        self.steps_per_epoch = steps_per_epoch
        self.kl_warmup_epochs = 20
        self.kl_rampup_epochs = 50
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 0.0005
        self.free_bits = 0.8

        self.z_std_ema = tf.Variable(1.0, trainable=False)

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_logvar = self.encoder([x, cond])
        z = sample_z(z_mean, z_logvar)
        x_hat = self.decoder([z, cond])
        return x_hat, z_mean, z_logvar

    def compute_kl_weight(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        warmup_done = tf.cast(step >= self.kl_warmup_steps, tf.float32)
        rampup_progress = (step - self.kl_warmup_steps) / self.kl_rampup_steps
        rampup_progress = tf.clip_by_value(rampup_progress, 0.0, 1.0)
        return self.kl_target * rampup_progress * warmup_done

    def compute_free_bits_kl(self, z_mean, z_logvar):
        kl_per_dim = -0.5 * (
            1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )
        kl_clamped = tf.maximum(kl_per_dim, self.free_bits)
        return tf.reduce_mean(kl_clamped)

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x)
            z = sample_z(z_mean, z_logvar)

            x_hat = self.decoder([z, cond])
            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))
            kl_free_bits = self.compute_free_bits_kl(z_mean, z_logvar)

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            kl_weight = self.compute_kl_weight()

            loss = (
                recon * recon_weight
                + stft_loss * STFT_weight
                + mel_loss * mel_weight
                + kl_free_bits * kl_weight
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)

        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "kl": kl_free_bits,
            "kl_weight": kl_weight,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
