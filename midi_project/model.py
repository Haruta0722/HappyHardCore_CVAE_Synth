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
recon_weight = 20.0
STFT_weight = 10.0
mel_weight = 8.0
kl_weight = 0.0005

channels = [
    (64, 5, 4),
    (128, 5, 4),
    (256, 5, 2),
    (512, 3, 2),
]

LATENT_STEPS = TIME_LENGTH // 64


class TimbreEnvelopeShaper(tf.keras.layers.Layer):
    """
    ★CVAE設計: [z, cond]からADSRパラメータを学習で生成
    """

    def __init__(self):
        super().__init__()

        # ★[z特徴, cond]を統合してADSRパラメータを生成（学習）
        self.adsr_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="sigmoid"),
            ],
            name="adsr_param_net",
        )

        # ★LFOパラメータも[z, cond]から学習
        self.lfo_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(2, activation="sigmoid"),
            ],
            name="lfo_param_net",
        )

    def call(self, base_envelope, timbre_weights, z_envelope_features):
        # ★重要: [z特徴, cond]を結合して学習
        combined_input = tf.concat(
            [z_envelope_features, timbre_weights], axis=-1
        )

        # ADSRパラメータを学習で生成
        adsr_params = self.adsr_param_net(combined_input)  # (B, 3)

        # パラメータを実際の値域に変換
        attack_speed = adsr_params[:, 0:1] * 25.0 + 1.0  # 1-26
        decay_rate = adsr_params[:, 1:2] * 20.0 + 0.5  # 0.5-20.5
        sustain_level = adsr_params[:, 2:3] * 0.9 + 0.05  # 0.05-0.95

        # LFOパラメータも学習で生成
        lfo_params = self.lfo_param_net(combined_input)  # (B, 2)
        lfo_rate = lfo_params[:, 0:1] * 8.0 + 1.0  # 1-9 Hz
        lfo_depth = lfo_params[:, 1:2] * 0.6  # 0-0.6

        # ADSR生成（これは微分可能な信号処理）
        time_length = tf.shape(base_envelope)[1]
        t = tf.cast(tf.range(time_length), tf.float32) / tf.cast(
            time_length, tf.float32
        )
        t = t[None, :]

        # ADSRエンベロープ（固定構造、パラメータは学習）
        attack_ratio = 0.08
        decay_ratio = 0.22

        attack_mask = tf.cast(t < attack_ratio, tf.float32)
        attack_curve = (t / (attack_ratio + 1e-6)) ** (1.0 / attack_speed)

        decay_end = attack_ratio + decay_ratio
        decay_mask = tf.cast((t >= attack_ratio) & (t < decay_end), tf.float32)
        decay_t = (t - attack_ratio) / (decay_ratio + 1e-6)
        decay_curve = 1.0 - (1.0 - sustain_level) * (
            1.0 - tf.exp(-decay_rate * decay_t)
        )

        release_mask = tf.cast(t >= decay_end, tf.float32)
        release_t = (t - decay_end) / (1.0 - decay_end + 1e-6)
        release_curve = sustain_level * tf.exp(-decay_rate * release_t)

        envelope_shape = (
            attack_mask * attack_curve
            + decay_mask * decay_curve
            + release_mask * release_curve
        )

        # LFO変調（微分可能な信号処理）
        t_seconds = t * WAV_LENGTH
        lfo_modulation = 1.0 + lfo_depth * tf.sin(
            2.0 * np.pi * lfo_rate * t_seconds
        )

        # base_envelopeと学習したenvelopeを組み合わせ
        final_envelope = base_envelope * 0.3 + envelope_shape * 0.7
        final_envelope = final_envelope * lfo_modulation

        return final_envelope


class TimbreShaper(tf.keras.layers.Layer):
    """
    ★CVAE設計: [z特徴, cond]から倍音プロファイルを学習
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # ★[base_amps特徴, cond]から倍音プロファイルを学習
        self.harmonic_profile_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="harmonic_profile_net",
        )

    def call(self, base_amps, timbre_weights):
        # base_ampsから時間平均で特徴抽出
        base_amps_mean = tf.reduce_mean(base_amps, axis=1)  # (B, H)

        # ★重要: [z由来の特徴, cond]を結合して学習
        combined_input = tf.concat([base_amps_mean, timbre_weights], axis=-1)

        # 倍音プロファイルを学習で生成
        harmonic_profile = self.harmonic_profile_net(combined_input)  # (B, H)
        harmonic_profile = harmonic_profile[:, None, :]  # (B, 1, H)

        # base_ampsと学習したプロファイルを組み合わせ
        shaped_amps = base_amps * (0.3 + harmonic_profile * 0.7)
        shaped_amps = tf.clip_by_value(shaped_amps, 0.0, 1.0)

        return shaped_amps


class GenerateHarmonicWaveTimeVarying(tf.keras.layers.Layer):
    """
    微分可能な加算合成（DDSP部分）
    """

    def __init__(self, sr=SR):
        super().__init__()
        self.sr = float(sr)

    def call(self, inputs):
        fundamental_freq, amplitudes_time, phases = inputs
        batch_size = tf.shape(fundamental_freq)[0]
        time_length = tf.shape(amplitudes_time)[1]
        num_harmonics = tf.shape(amplitudes_time)[2]

        t = tf.cast(tf.range(time_length), tf.float32) / float(self.sr)
        t = tf.reshape(t, [1, -1, 1])
        f0 = tf.reshape(fundamental_freq, [-1, 1, 1])
        harmonic_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
        harmonic_nums = tf.reshape(harmonic_nums, [1, 1, -1])

        omega = 2.0 * np.pi * f0 * harmonic_nums
        phas = tf.reshape(phases, [-1, 1, num_harmonics])
        harmonics = amplitudes_time * tf.sin(omega * t + phas)
        wave = tf.reduce_sum(harmonics, axis=-1)
        return wave


class EnvelopeNet(tf.keras.layers.Layer):
    """
    ★CVAE設計: zからエンベロープを生成（condは使わない）
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        # ★zのみから生成（condは後でDecoderで使う）
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 9, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    128, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    1, 3, padding="same", activation="sigmoid"
                ),
            ]
        )

        self.global_feature_net = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
            ]
        )

    def call(self, z, cond):
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        # グローバル特徴抽出（Shaperで使用）
        self.global_features = self.global_feature_net(z)

        # ★condをブロードキャスト（zと結合）
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        x = self.net(z_cond)
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

        return tf.squeeze(x, axis=-1), self.global_features


class HarmonicAmplitudeNet(tf.keras.layers.Layer):
    """
    ★CVAE設計: zから倍音振幅を生成（condは使わない）
    """

    def __init__(self, num_harmonics=NUM_HARMONICS, output_length=TIME_LENGTH):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.output_length = output_length

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 9, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    128, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    num_harmonics, 3, padding="same", activation="sigmoid"
                ),
            ]
        )

    def call(self, z, cond):
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        # ★condをブロードキャスト
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

        return amps


class NoiseGenerator(tf.keras.layers.Layer):
    """
    ★CVAE設計: [z, cond]からノイズエンベロープとフィルタ、量を学習
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        # zからノイズエンベロープ
        self.envelope_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    1, 3, padding="same", activation="sigmoid"
                ),
            ]
        )

        # zからノイズフィルタ特性
        self.filter_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    32, 9, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    16, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu"),
                tf.keras.layers.Conv1D(1, 3, padding="same", activation="tanh"),
            ]
        )

        # ★[cond]からノイズ量を学習
        self.noise_amount_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="noise_amount_net",
        )

    def call(self, z, cond):
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # zからノイズエンベロープ
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

        # zからフィルタされたノイズ
        batch_size = tf.shape(z)[0]
        random_noise = tf.random.normal([batch_size, self.output_length, 1])
        filtered_noise = self.filter_net(random_noise)

        # ★condからノイズ量を学習
        timbre = cond[:, 1:]  # [screech, acid, pluck]
        noise_amount = self.noise_amount_net(timbre)  # (B, 1)
        noise_amount = noise_amount[:, :, None]  # (B, 1, 1)

        output = noise_env * filtered_noise * noise_amount

        return tf.squeeze(output, axis=-1)


def build_encoder(latent_dim=LATENT_DIM):
    """
    ★CVAE設計: Encoderはaudioのみを入力（condを見ない）
    """
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))

    # ★重要: audioのみをエンコード（condを使わない）
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
        bias_initializer=tf.keras.initializers.Constant(-2.5),
    )(x)
    z_logvar = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -8.0, 2.0))(
        z_logvar
    )

    # cond_inはモデルの入力として必要だが、計算には使わない
    return tf.keras.Model([x_in], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    """
    ★CVAE設計: Decoderで初めてcondを使用
    """
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # ★zから基本パラメータを生成（condも渡すが、内部で結合）
    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    base_harmonic_amps = harmonic_amp_net(z_in, cond)

    # ★Shaperで[z特徴, cond]から学習
    timbre = tf.keras.layers.Lambda(lambda c: c[:, 1:])(cond)
    timbre_shaper = TimbreShaper(num_harmonics=NUM_HARMONICS)
    shaped_harmonic_amps = timbre_shaper(base_harmonic_amps, timbre)

    # ★zからエンベロープ生成
    envelope_net = EnvelopeNet()
    base_envelope, z_envelope_features = envelope_net(z_in, cond)

    # ★Shaperで[z特徴, cond]から学習
    envelope_shaper = TimbreEnvelopeShaper()
    envelope = envelope_shaper(base_envelope, timbre, z_envelope_features)

    # ★zとcondからノイズ生成
    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)

    # ピッチから基本周波数（これは明示的）
    pitch = tf.keras.layers.Lambda(lambda c: c[:, 0])(cond)
    fundamental_freq = tf.keras.layers.Lambda(
        lambda p: 440.0 * tf.pow(2.0, ((p * 35.0 + 36.0) - 69.0) / 12.0)
    )(pitch)

    # 倍音波形生成（微分可能な加算合成）
    initial_amps = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
        shaped_harmonic_amps
    )
    phases = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(initial_amps)

    harmonic_wave_layer = GenerateHarmonicWaveTimeVarying()
    harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, shaped_harmonic_amps, phases]
    )

    # 最終合成
    output = harmonic_wave * envelope + noise
    output = tf.keras.layers.Activation("tanh")(output)
    output = tf.keras.layers.Lambda(lambda x: x[:, :, None])(output)

    return tf.keras.Model([z_in, cond], output, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)

        self.steps_per_epoch = steps_per_epoch
        # ★CVAE設計: KLパラメータ調整
        self.kl_warmup_epochs = 35
        self.kl_rampup_epochs = 70
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 0.0002
        self.free_bits = 0.4
        self.z_std_ema = tf.Variable(1.0, trainable=False)
        self.best_recon = tf.Variable(float("inf"), trainable=False)

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_logvar = self.encoder(x)
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
                recon * 20.0
                + stft_loss * 10.0
                + mel_loss * 8.0
                + kl_free_bits * kl_weight
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)

        if recon < self.best_recon:
            self.best_recon.assign(recon)

        return {
            "loss": loss,
            "recon": recon,
            "best_recon": self.best_recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "kl": kl_free_bits,
            "kl_weight": kl_weight,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
