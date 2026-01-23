import tensorflow as tf
from loss import Loss
import numpy as np

SR = 48000
COND_DIM = 3 + 1
LATENT_DIM = 64
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)
NUM_HARMONICS = 32

channels = [
    (64, 5, 4),
    (128, 5, 4),
    (256, 5, 2),
    (512, 3, 2),
]

LATENT_STEPS = TIME_LENGTH // 64


# ========================================
# LearnablePrototypes
# ========================================
class LearnablePrototypes(tf.keras.layers.Layer):
    def __init__(self, latent_dim=LATENT_DIM, latent_steps=LATENT_STEPS):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_steps = latent_steps

        self.prototypes = self.add_weight(
            name="timbre_prototypes",
            shape=(3, latent_steps, latent_dim),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.1
            ),
            trainable=True,
        )

        self.pitch_modulation = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(latent_steps * latent_dim),
                tf.keras.layers.Reshape((latent_steps, latent_dim)),
            ],
            name="pitch_modulation",
        )

    def call(self, cond):
        batch_size = tf.shape(cond)[0]
        pitch = cond[:, 0:1]
        timbre = cond[:, 1:]

        timbre_expanded = tf.expand_dims(timbre, axis=1)
        base_z = tf.einsum("bnt,tsd->bsd", timbre_expanded, self.prototypes)

        pitch_mod = self.pitch_modulation(pitch)
        z = base_z + pitch_mod * 0.2

        return z


# ========================================
# HighFrequencyEmphasis
# ========================================
class HighFrequencyEmphasis(tf.keras.layers.Layer):
    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        self.hf_emphasis_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="hf_emphasis",
        )

    def call(self, amplitudes, timbre):
        hf_profile = self.hf_emphasis_net(timbre)

        harmonic_indices = tf.range(1, self.num_harmonics + 1, dtype=tf.float32)
        hf_curve = tf.pow(harmonic_indices / float(self.num_harmonics), 0.3)
        hf_curve = tf.reshape(hf_curve, [1, 1, self.num_harmonics])

        hf_profile_expanded = hf_profile[:, None, :]
        emphasis_factor = 1.0 + hf_profile_expanded * hf_curve * 2.0
        emphasized_amps = amplitudes * emphasis_factor

        return emphasized_amps


# ========================================
# TimbreEnvelopeShaper
# ========================================
class TimbreEnvelopeShaper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.adsr_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="sigmoid"),
            ],
            name="adsr_param_net",
        )

        self.lfo_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(2, activation="sigmoid"),
            ],
            name="lfo_param_net",
        )

    def call(self, base_envelope, timbre_weights, z_envelope_features):
        combined_input = tf.concat(
            [z_envelope_features, timbre_weights], axis=-1
        )

        adsr_params = self.adsr_param_net(combined_input)
        attack_speed = adsr_params[:, 0:1] * 25.0 + 1.0
        decay_rate = adsr_params[:, 1:2] * 20.0 + 0.5
        sustain_level = adsr_params[:, 2:3] * 0.9 + 0.05

        lfo_params = self.lfo_param_net(combined_input)
        lfo_rate = lfo_params[:, 0:1] * 8.0 + 1.0
        lfo_depth = lfo_params[:, 1:2] * 0.6

        time_length = tf.shape(base_envelope)[1]
        t = tf.cast(tf.range(time_length), tf.float32) / tf.cast(
            time_length, tf.float32
        )
        t = t[None, :]

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

        t_seconds = t * WAV_LENGTH
        lfo_modulation = 1.0 + lfo_depth * tf.sin(
            2.0 * np.pi * lfo_rate * t_seconds
        )

        final_envelope = base_envelope * 0.3 + envelope_shape * 0.7
        final_envelope = final_envelope * lfo_modulation

        return final_envelope


# ========================================
# TimbreShaper
# ========================================
class TimbreShaper(tf.keras.layers.Layer):
    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        self.harmonic_profile_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation="sigmoid"),
            ],
            name="harmonic_profile_net",
        )

    def call(self, base_amps, timbre_weights):
        base_amps_mean = tf.reduce_mean(base_amps, axis=1)
        combined_input = tf.concat([base_amps_mean, timbre_weights], axis=-1)

        harmonic_profile = self.harmonic_profile_net(combined_input)
        harmonic_profile = harmonic_profile[:, None, :]

        shaped_amps = base_amps * (0.5 + harmonic_profile * 0.5)
        shaped_amps = tf.clip_by_value(shaped_amps, 0.0, 1.0)

        return shaped_amps


# ========================================
# GenerateHarmonicWaveTimeVarying
# ========================================
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


# ========================================
# EnvelopeNet
# ========================================
class EnvelopeNet(tf.keras.layers.Layer):
    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
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


# ========================================
# HarmonicAmplitudeNet
# ========================================
class HarmonicAmplitudeNet(tf.keras.layers.Layer):
    def __init__(self, num_harmonics=NUM_HARMONICS, output_length=TIME_LENGTH):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.output_length = output_length

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    256, 9, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    256, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    128, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    num_harmonics, 3, padding="same", activation="sigmoid"
                ),
            ]
        )

        self.hf_emphasis = HighFrequencyEmphasis(num_harmonics)

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

        timbre = cond[:, 1:]
        amps = self.hf_emphasis(amps, timbre)

        return amps


# ========================================
# NoiseGenerator
# ========================================
class NoiseGenerator(tf.keras.layers.Layer):
    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

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

        timbre = cond[:, 1:]
        noise_amount = self.noise_amount_net(timbre)
        noise_amount = noise_amount * 0.02
        noise_amount = noise_amount[:, :, None]

        output = noise_env * filtered_noise * noise_amount

        return tf.squeeze(output, axis=-1)


# ========================================
# Encoder
# ========================================
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
        bias_initializer=tf.keras.initializers.Constant(-2.0),
    )(x)
    z_logvar = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -8.0, 2.0))(
        z_logvar
    )

    return tf.keras.Model([x_in], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


# ========================================
# Decoder（★完全実装版）
# ========================================
def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # 倍音振幅生成
    harmonic_amp_net = HarmonicAmplitudeNet(num_harmonics=NUM_HARMONICS)
    base_harmonic_amps = harmonic_amp_net(z_in, cond)

    # 音色による倍音シェイピング
    timbre = tf.keras.layers.Lambda(lambda c: c[:, 1:])(cond)
    timbre_shaper = TimbreShaper(num_harmonics=NUM_HARMONICS)
    shaped_harmonic_amps = timbre_shaper(base_harmonic_amps, timbre)

    # エンベロープ生成
    envelope_net = EnvelopeNet()
    base_envelope, z_envelope_features = envelope_net(z_in, cond)

    # 音色によるエンベロープシェイピング
    envelope_shaper = TimbreEnvelopeShaper()
    envelope = envelope_shaper(base_envelope, timbre, z_envelope_features)

    # ノイズ生成
    noise_gen = NoiseGenerator()
    noise = noise_gen(z_in, cond)

    # ピッチから基本周波数を計算
    pitch = tf.keras.layers.Lambda(lambda c: c[:, 0])(cond)
    fundamental_freq = tf.keras.layers.Lambda(
        lambda p: 440.0 * tf.pow(2.0, ((p * 35.0 + 36.0) - 69.0) / 12.0)
    )(pitch)

    # 位相初期化
    initial_amps = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(
        shaped_harmonic_amps
    )
    phases = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(initial_amps)

    # 倍音波形生成
    harmonic_wave_layer = GenerateHarmonicWaveTimeVarying()
    harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, shaped_harmonic_amps, phases]
    )

    # 最終合成
    output = harmonic_wave * envelope + noise
    output = tf.keras.layers.Activation("tanh")(output)
    output = tf.keras.layers.Lambda(lambda x: x[:, :, None])(output)

    return tf.keras.Model([z_in, cond], output, name="decoder")


# ========================================
# TimeWiseCVAE
# ========================================
class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
        self.prototypes = LearnablePrototypes(latent_dim, LATENT_STEPS)

        self.steps_per_epoch = steps_per_epoch
        self.kl_warmup_epochs = 30
        self.kl_rampup_epochs = 100
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 0.0001
        self.free_bits = 0.1

        self.z_std_ema = tf.Variable(1.0, trainable=False)
        self.best_recon = tf.Variable(float("inf"), trainable=False)

    def call(self, inputs, training=None):
        x, cond = inputs
        z_mean, z_logvar = self.encoder(x, training=training)
        z = sample_z(z_mean, z_logvar)
        x_hat = self.decoder([z, cond], training=training)
        return x_hat, z_mean, z_logvar

    def generate(self, cond):
        """推論用"""
        z = self.prototypes(cond)
        x_hat = self.decoder([z, cond], training=False)
        return x_hat

    def compute_kl_weight(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        warmup_done = tf.cast(step >= self.kl_warmup_steps, tf.float32)
        rampup_progress = (step - self.kl_warmup_steps) / self.kl_rampup_steps
        rampup_progress = tf.clip_by_value(rampup_progress, 0.0, 1.0)
        return self.kl_target * rampup_progress * warmup_done

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x, training=True)
            z = sample_z(z_mean, z_logvar)
            z_prototype = self.prototypes(cond)

            x_hat = self.decoder([z, cond], training=True)
            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            kl_per_dim = -0.5 * (
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            kl_divergence = tf.reduce_mean(
                tf.maximum(kl_per_dim, self.free_bits)
            )

            kl_weight = self.compute_kl_weight()

            prototype_loss = tf.reduce_mean(
                tf.square(z_mean - tf.stop_gradient(z_prototype))
            )

            loss = (
                recon * 25.0
                + stft_loss * 12.0
                + mel_loss * 10.0
                + kl_divergence * kl_weight
                + prototype_loss * 1.0
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)
        self.best_recon.assign(tf.minimum(self.best_recon, recon))

        return {
            "loss": loss,
            "recon": recon,
            "best_recon": self.best_recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "kl": kl_divergence,
            "kl_weight": kl_weight,
            "prototype_loss": prototype_loss,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
