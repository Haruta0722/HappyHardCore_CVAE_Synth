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

channels = [
    (64, 5, 4),
    (128, 5, 4),
    (256, 5, 2),
    (512, 3, 2),
]

LATENT_STEPS = TIME_LENGTH // 64


class TimbreShaper(tf.keras.layers.Layer):
    """
    ★新機能: 音色ごとに倍音バランスを明示的に変える
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # 各音色の倍音プロファイルを学習
        self.screech_profile = tf.keras.layers.Dense(
            num_harmonics, activation="sigmoid", name="screech_profile"
        )
        self.acid_profile = tf.keras.layers.Dense(
            num_harmonics, activation="sigmoid", name="acid_profile"
        )
        self.pluck_profile = tf.keras.layers.Dense(
            num_harmonics, activation="sigmoid", name="pluck_profile"
        )

    def call(self, base_amps, timbre_weights):
        """
        Args:
            base_amps: (B, T, H) - 基本的な倍音振幅
            timbre_weights: (B, 3) - [screech, acid, pluck]

        Returns:
            shaped_amps: (B, T, H) - 音色で調整された倍音振幅
        """
        # 各音色のプロファイル
        screech_w = timbre_weights[:, 0:1]  # (B, 1)
        acid_w = timbre_weights[:, 1:2]
        pluck_w = timbre_weights[:, 2:3]

        # プロファイルを生成（重みから）
        screech_prof = self.screech_profile(screech_w)  # (B, H)
        acid_prof = self.acid_profile(acid_w)
        pluck_prof = self.pluck_profile(pluck_w)

        # 重み付き合成
        combined_profile = (
            screech_prof * screech_w + acid_prof * acid_w + pluck_prof * pluck_w
        )  # (B, H)

        # 時間方向に展開
        combined_profile = combined_profile[:, None, :]  # (B, 1, H)

        # base_ampsに音色プロファイルを適用
        shaped_amps = base_amps * (0.5 + combined_profile * 1.5)

        return shaped_amps


class GenerateHarmonicWaveTimeVarying(tf.keras.layers.Layer):
    """
    ★改善: 時間変化する倍音振幅に対応
    """

    def __init__(self, sr=SR):
        super().__init__()
        self.sr = sr

    def call(self, inputs):
        fundamental_freq, amplitudes_time, phases = inputs
        # amplitudes_time: (B, T, H) - 時間変化する倍音振幅
        # phases: (B, H) - 初期位相

        batch_size = tf.shape(fundamental_freq)[0]
        time_length = tf.shape(amplitudes_time)[1]
        num_harmonics = tf.shape(amplitudes_time)[2]

        # 時間軸
        t = tf.cast(tf.range(time_length), tf.float32) / float(self.sr)
        t = tf.reshape(t, [1, -1, 1])  # (1, T, 1)

        # 基本周波数
        f0 = tf.reshape(fundamental_freq, [-1, 1, 1])  # (B, 1, 1)

        # 倍音番号
        harmonic_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
        harmonic_nums = tf.reshape(harmonic_nums, [1, 1, -1])  # (1, 1, H)

        # 角周波数
        omega = 2.0 * np.pi * f0 * harmonic_nums  # (B, 1, H)

        # 位相を展開
        phas = tf.reshape(phases, [-1, 1, num_harmonics])  # (B, 1, H)

        # ★重要: 時間変化する振幅を使用
        # amplitudes_time は既に (B, T, H)
        harmonics = amplitudes_time * tf.sin(omega * t + phas)

        wave = tf.reduce_sum(harmonics, axis=-1)  # (B, T)

        return wave


class EnvelopeNet(tf.keras.layers.Layer):
    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

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
        x = self.net(z)

        x = tf.keras.layers.Lambda(
            lambda v: tf.squeeze(
                tf.image.resize(
                    tf.expand_dims(v, axis=2),
                    [self.output_length, 1],
                    method="bilinear",
                ),
                axis=2,
            )
        )(x)

        return tf.squeeze(x, axis=-1)


class HarmonicAmplitudeNet(tf.keras.layers.Layer):
    def __init__(self, num_harmonics=NUM_HARMONICS, output_length=TIME_LENGTH):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.output_length = output_length

        # ★改善: より深いネットワークで複雑な倍音パターンを学習
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 5, padding="same", activation="relu"
                ),
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

        amps = tf.keras.layers.Lambda(
            lambda v: tf.squeeze(
                tf.image.resize(
                    tf.expand_dims(v, axis=2),
                    [self.output_length, 1],
                    method="bilinear",
                ),
                axis=2,
            )
        )(amps)

        return amps  # (B, T, H)


class NoiseGenerator(tf.keras.layers.Layer):
    """
    ★改善: より表現力のあるノイズ生成
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        # ノイズのエンベロープを生成
        self.envelope_net = tf.keras.Sequential(
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

        # ノイズのフィルタリング（周波数特性）
        self.filter_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    16, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(1, 5, padding="same", activation="tanh"),
            ]
        )

    def call(self, z, cond):
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # ノイズのエンベロープ
        noise_env = self.envelope_net(z_cond)
        noise_env = tf.keras.layers.Lambda(
            lambda v: tf.squeeze(
                tf.image.resize(
                    tf.expand_dims(v, axis=2),
                    [self.output_length, 1],
                    method="bilinear",
                ),
                axis=2,
            )
        )(noise_env)

        # ホワイトノイズ生成
        batch_size = tf.shape(z)[0]
        random_noise = tf.random.normal([batch_size, self.output_length, 1])

        # ノイズのフィルタリング
        filtered_noise = self.filter_net(random_noise)

        # エンベロープを適用
        output = noise_env * filtered_noise

        return tf.squeeze(output, axis=-1)


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

    # 倍音振幅生成（時間変化あり）
    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    base_harmonic_amps = harmonic_amp_net(z_in, cond)  # (B, T, H)

    # ★新機能: 音色による倍音シェーピング
    timbre = tf.keras.layers.Lambda(lambda c: c[:, 1:])(cond)
    timbre_shaper = TimbreShaper(num_harmonics=NUM_HARMONICS)
    shaped_harmonic_amps = timbre_shaper(
        base_harmonic_amps, timbre
    )  # (B, T, H)

    # エンベロープ生成
    envelope_net = EnvelopeNet()
    envelope = envelope_net(z_in)  # (B, T)

    # ノイズ生成
    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)  # (B, T)

    # 基本周波数
    pitch = tf.keras.layers.Lambda(lambda c: c[:, 0])(cond)
    fundamental_freq = tf.keras.layers.Lambda(
        lambda p: 440.0 * tf.pow(2.0, ((p * 35.0 + 36.0) - 69.0) / 12.0)
    )(pitch)

    # 初期位相（時間軸の最初の倍音振幅から推定）
    initial_amps = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
        shaped_harmonic_amps
    )
    phases = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(initial_amps)

    # ★改善: 時間変化する倍音合成
    harmonic_wave_layer = GenerateHarmonicWaveTimeVarying()
    harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, shaped_harmonic_amps, phases]
    )

    # ★改善: 音色によって倍音/ノイズの比率を動的に変える
    screech_w = tf.keras.layers.Lambda(lambda c: c[:, 1:2])(cond)
    acid_w = tf.keras.layers.Lambda(lambda c: c[:, 2:3])(cond)
    pluck_w = tf.keras.layers.Lambda(lambda c: c[:, 3:4])(cond)

    # screech: ノイズ強め (0.5), 倍音も強い
    # acid: バランス (0.2)
    # pluck: ノイズ弱め (0.05), 倍音中心
    noise_ratio = screech_w * 0.5 + acid_w * 0.2 + pluck_w * 0.05
    harmonic_ratio = (
        1.0 - noise_ratio * 0.5
    )  # ノイズが増えても倍音は減らしすぎない

    # 最終合成
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
