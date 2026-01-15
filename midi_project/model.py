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


class TimbreEnvelopeShaper(tf.keras.layers.Layer):
    """
    ★改善: condから直接音色特性を学習し、zは変調のみに使用
    """

    def __init__(self):
        super().__init__()

        # 各音色の基本的な減衰パターン（condから直接生成）
        self.timbre_decay_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(3, activation="sigmoid"),
            ]
        )

        # zによる変調量（小さな調整のみ）
        self.z_modulation_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(3, activation="tanh"),  # -1~1の変調
            ]
        )

        # acid用: うねり（LFO）パラメータ
        self.lfo_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(2, activation="sigmoid"),
            ]
        )

    def call(self, base_envelope, timbre_weights, z_envelope_features):
        # ★改善1: 音色から直接エンベロープパラメータを生成
        env_params = self.timbre_decay_net(timbre_weights)

        # pluckの影響を強化: pluck_weightが高いほど急速減衰
        pluck_w = timbre_weights[:, 2:3]
        attack_speed = env_params[:, 0:1] * 15.0 + 5.0 + pluck_w * 20.0
        decay_rate = env_params[:, 1:2] * 10.0 + 5.0 + pluck_w * 15.0
        sustain_level = env_params[:, 2:3] * 0.5 + 0.2 - pluck_w * 0.3

        # screechの影響: 長い持続
        screech_w = timbre_weights[:, 0:1]
        decay_rate = decay_rate - screech_w * 8.0  # 減衰を遅く
        sustain_level = sustain_level + screech_w * 0.4  # サステインを高く

        # ★改善2: zによる小さな変調（±20%程度）
        z_mod = self.z_modulation_net(z_envelope_features) * 0.2
        attack_speed = attack_speed * (1.0 + z_mod[:, 0:1])
        decay_rate = decay_rate * (1.0 + z_mod[:, 1:2])
        sustain_level = tf.clip_by_value(
            sustain_level * (1.0 + z_mod[:, 2:3]), 0.05, 0.95
        )

        # acid用: LFOパラメータ
        acid_w = timbre_weights[:, 1:2]
        lfo_params = self.lfo_net(
            tf.concat([acid_w, z_envelope_features[:, :8]], axis=-1)
        )
        lfo_rate = lfo_params[:, 0:1] * 6.0 + 3.0  # 3-9 Hz
        lfo_depth = lfo_params[:, 1:2] * 0.5 * acid_w  # acidの重みに比例

        # 時間軸を生成
        time_length = tf.shape(base_envelope)[1]
        t = tf.cast(tf.range(time_length), tf.float32) / tf.cast(
            time_length, tf.float32
        )
        t = t[None, :]

        # ★改善3: より明確なADSRエンベロープ
        # pluckの場合: attack 5%, decay 30%, release 65%
        # それ以外: attack 10%, decay 20%, sustain+release 70%
        attack_ratio = 0.05 + (1.0 - pluck_w) * 0.05
        decay_ratio = 0.30 * pluck_w + 0.20 * (1.0 - pluck_w)

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
        # pluckの場合は速い減衰、それ以外は緩やか
        release_decay = decay_rate * (0.5 + pluck_w * 2.0)
        release_curve = sustain_level * tf.exp(-release_decay * release_t)

        envelope_shape = (
            attack_mask * attack_curve
            + decay_mask * decay_curve
            + release_mask * release_curve
        )

        # ★改善4: acid用のLFO変調を強化
        t_seconds = t * WAV_LENGTH
        lfo_modulation = 1.0 + lfo_depth * tf.sin(
            2.0 * np.pi * lfo_rate * t_seconds
        )
        modulated_envelope = envelope_shape * lfo_modulation

        # base_envelopeとの混合比を調整（音色特性を優先）
        shaped_envelope = (
            envelope_shape * 0.7 + base_envelope * modulated_envelope * 0.3
        )

        return shaped_envelope


class TimbreShaper(tf.keras.layers.Layer):
    """
    ★改善: 音色ごとの明確な倍音プロファイルを定義
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # 各音色の固定的な倍音特性を学習
        self.screech_profile = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="screech_profile",
        )

        self.acid_profile = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="acid_profile",
        )

        self.pluck_profile = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="pluck_profile",
        )

    def call(self, base_amps, timbre_weights):
        screech_w = timbre_weights[:, 0:1]
        acid_w = timbre_weights[:, 1:2]
        pluck_w = timbre_weights[:, 2:3]

        # ★改善: 各音色のプロファイルを音色weightから直接生成
        screech_prof = self.screech_profile(screech_w)  # (B, H)
        acid_prof = self.acid_profile(acid_w)
        pluck_prof = self.pluck_profile(pluck_w)

        # ★screech: 高音域にノイジーな成分（但し過剰ではない）
        harmonic_indices = tf.range(1, self.num_harmonics + 1, dtype=tf.float32)
        high_freq_emphasis = tf.pow(harmonic_indices / self.num_harmonics, 1.5)
        high_freq_emphasis = tf.reshape(high_freq_emphasis, [1, -1])
        screech_prof = screech_prof * (0.3 + high_freq_emphasis * 0.7)

        # ★acid: 中音域（3-12倍音）を強調
        mid_freq_mask = tf.exp(-0.5 * ((harmonic_indices - 7.0) / 4.0) ** 2)
        mid_freq_mask = tf.reshape(mid_freq_mask, [1, -1])
        acid_prof = acid_prof * (0.5 + mid_freq_mask * 1.5)

        # ★pluck: 低音域重視、高音域減衰
        low_freq_emphasis = tf.exp(-harmonic_indices / 10.0)
        low_freq_emphasis = tf.reshape(low_freq_emphasis, [1, -1])
        pluck_prof = pluck_prof * (0.2 + low_freq_emphasis * 0.8)

        # 重み付き合成
        combined_profile = (
            screech_prof * screech_w + acid_prof * acid_w + pluck_prof * pluck_w
        )
        combined_profile = combined_profile[:, None, :]  # (B, 1, H)

        # base_ampsとの混合（音色特性を強く反映）
        shaped_amps = base_amps * 0.3 + combined_profile * 0.7
        shaped_amps = tf.clip_by_value(shaped_amps, 0.0, 1.0)

        return shaped_amps


class GenerateHarmonicWaveTimeVarying(tf.keras.layers.Layer):
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
    ★改善: zの影響を適度に抑える
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        self.net = tf.keras.Sequential(
            [
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
                tf.keras.layers.Dense(32, activation="relu"),
            ]
        )

    def call(self, z, cond):
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        self.global_features = self.global_feature_net(z)

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

        return amps


class NoiseGenerator(tf.keras.layers.Layer):
    """
    ★改善: screech用のノイズを制御可能に
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

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

        batch_size = tf.shape(z)[0]
        random_noise = tf.random.normal([batch_size, self.output_length, 1])
        filtered_noise = self.filter_net(random_noise)
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

    # ★改善: z_logvarの初期化を調整（分散が大きすぎないように）
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-2.0),  # -3.0から-2.0に
    )(x)
    z_logvar = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -8.0, 2.0)  # 範囲を狭める
    )(z_logvar)

    return tf.keras.Model([x_in, cond_in], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    base_harmonic_amps = harmonic_amp_net(z_in, cond)

    timbre = tf.keras.layers.Lambda(lambda c: c[:, 1:])(cond)
    timbre_shaper = TimbreShaper(num_harmonics=NUM_HARMONICS)
    shaped_harmonic_amps = timbre_shaper(base_harmonic_amps, timbre)

    envelope_net = EnvelopeNet()
    base_envelope, z_envelope_features = envelope_net(z_in, cond)

    envelope_shaper = TimbreEnvelopeShaper()
    envelope = envelope_shaper(base_envelope, timbre, z_envelope_features)

    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)

    pitch = tf.keras.layers.Lambda(lambda c: c[:, 0])(cond)
    fundamental_freq = tf.keras.layers.Lambda(
        lambda p: 440.0 * tf.pow(2.0, ((p * 35.0 + 36.0) - 69.0) / 12.0)
    )(pitch)

    initial_amps = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
        shaped_harmonic_amps
    )
    phases = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(initial_amps)

    harmonic_wave_layer = GenerateHarmonicWaveTimeVarying()
    harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, shaped_harmonic_amps, phases]
    )

    screech_w = tf.keras.layers.Lambda(lambda c: c[:, 1:2])(cond)
    acid_w = tf.keras.layers.Lambda(lambda c: c[:, 2:3])(cond)
    pluck_w = tf.keras.layers.Lambda(lambda c: c[:, 3:4])(cond)

    # ★改善: ノイズ比率を調整（screechのノイズを減らす）
    noise_ratio = (
        screech_w * 0.3 + acid_w * 0.15 + pluck_w * 0.02
    )  # screechを0.6→0.3に
    harmonic_ratio = 1.0 - noise_ratio * 0.2  # 倍音を優先

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
        self.kl_warmup_epochs = 30  # 20→30に延長
        self.kl_rampup_epochs = 60  # 50→60に延長
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 0.0003  # 0.0005→0.0003に減少
        self.free_bits = 1.0  # 0.8→1.0に増加
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
