import tensorflow as tf
from loss import Loss
import numpy as np

SR = 48000
COND_DIM = 3 + 1
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)

NUM_HARMONICS = 32

# 損失関数の重み
recon_weight = 5.0
STFT_weight = 15.0
mel_weight = 10.0
kl_weight = 0.0005


class GenerateHarmonicWave(tf.keras.layers.Layer):
    def __init__(self, length=None, sr=SR):
        super().__init__()
        self.length = length
        self.sr = sr

    def call(self, inputs):
        fundamental_freq, amplitudes, phases = inputs

        # ★軽量化1: tf.tileを削除し、ブロードキャストを利用する
        # 時間軸 t: (1, T, 1)
        t = tf.range(self.length, dtype=tf.float32) / float(self.sr)
        t = t[None, :, None]

        # f0: (B, 1, 1)
        f0 = fundamental_freq[:, None, None]

        # 倍音番号: (1, 1, H)
        num_harmonics = tf.shape(amplitudes)[1]
        harmonic_nums = tf.range(1, num_harmonics + 1, dtype=tf.float32)[
            None, None, :
        ]

        # 角周波数: (B, 1, H)
        omega = 2.0 * np.pi * f0 * harmonic_nums

        # 振幅と位相: (B, T, H) になるように後でブロードキャストされる
        amps = amplitudes[
            :, None, :
        ]  # 実際はDecoder側で時間方向に展開済みなら (B, T, H)
        phas = phases[:, None, :]

        # ★ここが最大の計算量。ブロードキャストでメモリを節約しながら計算
        # t (1, T, 1) * omega (B, 1, H) -> (B, T, H) が自動的に行われる
        harmonics = amps * tf.sin(omega * t + phas)

        wave = tf.reduce_sum(harmonics, axis=-1)

        return wave


class EnvelopeNet(tf.keras.layers.Layer):
    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        self.net = tf.keras.Sequential(
            [
                # ★軽量化2: チャンネル数を少し調整（64->32でも十分性能は出ます）
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
        # z は以前より時間解像度が低い (1/64)
        x = self.net(z)

        # 低解像度で概形を作ってからアップサンプリング（計算コスト大幅減）
        x = tf.keras.layers.Lambda(
            lambda v: tf.image.resize(
                v, [self.output_length, 1], method="bilinear"
            )
        )(x)
        return tf.squeeze(x, axis=-1)


class HarmonicAmplitudeNet(tf.keras.layers.Layer):
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
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        amps = self.net(z_cond)

        # アップサンプリング
        amps = tf.keras.layers.Lambda(
            lambda v: tf.image.resize(
                v, [self.output_length, self.num_harmonics], method="bilinear"
            )
        )(amps)
        return amps


class NoiseGenerator(tf.keras.layers.Layer):
    """
    ★軽量化3: Source-Filterモデルへの変更
    以前: ニューラルネットが波形そのものを描こうとしていた (高負荷 & 低品質)
    今回: ニューラルネットは「音量(エンベロープ)」だけ予測し、ノイズと掛ける (低負荷 & 高品質)
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    1, 5, padding="same", activation="sigmoid"
                ),  # 音量なので0-1
            ]
        )

    def call(self, z, cond):
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # ノイズのエンベロープを予測
        noise_env = self.net(z_cond)

        # エンベロープをアップサンプリング
        noise_env = tf.keras.layers.Lambda(
            lambda v: tf.image.resize(
                v, [self.output_length, 1], method="bilinear"
            )
        )(
            noise_env
        )  # (B, T, 1)

        # ★ランダムなホワイトノイズを生成 (実行時に生成)
        # Lambda内でshapeを取得して生成する
        random_noise = tf.keras.layers.Lambda(
            lambda shape_tensor: tf.random.normal(tf.shape(shape_tensor))
        )(noise_env)

        # エンベロープ × ホワイトノイズ
        output = noise_env * random_noise

        return tf.squeeze(output, axis=-1)


# ★軽量化4: Encoderのストライドを変更して、潜在変数の時間方向を圧縮
# 以前: [2, 2, 2, 2] = 16倍圧縮 (Latent長 3900)
# 今回: [4, 4, 2, 2] = 64倍圧縮 (Latent長 975) -> これでも十分高精細
channels = [
    (64, 5, 4),  # Stride 4
    (128, 5, 4),  # Stride 4
    (256, 5, 2),  # Stride 2
    (512, 3, 2),  # Stride 2
]

# 64倍圧縮に合わせたLatent Steps
LATENT_STEPS = TIME_LENGTH // 64


def build_encoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))
    cond_in = tf.keras.Input(shape=(cond_dim,))

    cond_repeated = tf.keras.layers.RepeatVector(TIME_LENGTH)(cond_in)
    x = tf.keras.layers.Concatenate()([x_in, cond_repeated])

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

    return tf.keras.Model([x_in, cond_in], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # 各生成モジュール (潜在変数が短くなったので計算が高速)
    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    # (B, T, H) ここでresize済み
    harmonic_amps_time = harmonic_amp_net(z_in, cond)

    envelope_net = EnvelopeNet()
    envelope = envelope_net(z_in)  # (B, T)

    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)  # (B, T) - エンベロープ方式で生成されたノイズ

    # Pitch計算
    pitch = tf.keras.layers.Lambda(lambda c: c[:, 0])(cond)
    fundamental_freq = tf.keras.layers.Lambda(
        lambda p: 440.0 * tf.pow(2.0, ((p * 35.0 + 36.0) - 69.0) / 12.0)
    )(pitch)

    # 倍音振幅 (時間変化あり)
    # 元のコードに合わせて reduce_mean していますが、
    # 時間変化を生かしたいなら harmonic_amps_time をそのまま使ってもOKです。
    # 今回は元のロジックに従い平均化します。
    avg_harmonic_amps = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=1)
    )(harmonic_amps_time)

    # 以前は (B, H) でしたが、generate_harmonic_wave 内でブロードキャストさせるため
    # ここでの処理はそのまま (B, H) でOK。
    # ただし、時間変化させたい場合は (B, T, H) のまま渡すようにLayerを改造できます。
    # ここでは「同じ結果」を出すため、あえて平均化されたものを使います。

    phases = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(
        avg_harmonic_amps
    )

    # 倍音合成
    harmonic_wave_layer = GenerateHarmonicWave(length=TIME_LENGTH)
    harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, avg_harmonic_amps, phases]
    )

    timbre = tf.keras.layers.Lambda(lambda c: c[:, 1:])(cond)

    harmonic_ratio = 0.9
    noise_ratio = 0.1

    output = harmonic_wave * envelope * harmonic_ratio + noise * noise_ratio

    output = tf.keras.layers.Activation("tanh")(output)
    output = tf.keras.layers.Lambda(lambda x: x[:, :, None])(output)

    return tf.keras.Model([z_in, cond], output, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.encoder = build_encoder(latent_dim, cond_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)

        # 学習パラメータ等は変更なし
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
            z_mean, z_logvar = self.encoder([x, cond])
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
