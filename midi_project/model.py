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
# DDSP Module 1: Learnable Unison/Detune
# ========================================
class DDSPUnisonDetuneLayer(tf.keras.layers.Layer):
    """
    データから学習するUnison/Detune:
    - voice数、detune量、spread量を全て学習
    - カテゴリ分岐なし
    """

    def __init__(self, max_voices=16, sr=SR):
        super().__init__()
        self.max_voices = max_voices
        self.sr = sr

        # z + condから全パラメータを学習
        self.param_predictor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # [num_voices, detune_cents, spread, phase_mod_depth]
                tf.keras.layers.Dense(4, activation=None),
            ],
            name="unison_params",
        )

        # 各voiceの時間変化するゲイン
        self.voice_gain_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    max_voices, 3, padding="same", activation=None
                ),
            ],
            name="voice_gains",
        )

    def call(self, base_signal, z, cond, fundamental_freq):
        """
        Args:
            base_signal: [batch, time_length]
            z: [batch, latent_steps, latent_dim]
            cond: [batch, cond_dim]
            fundamental_freq: [batch]
        """
        batch_size = tf.shape(base_signal)[0]

        # zの全体特徴を抽出
        z_global = tf.reduce_mean(z, axis=1)  # [batch, latent_dim]
        combined_input = tf.concat([z_global, cond], axis=-1)

        # パラメータ予測
        params = self.param_predictor(combined_input)  # [batch, 4]

        # 各パラメータを適切な範囲に変換
        num_voices_logit = params[:, 0:1]  # [-inf, inf]
        num_voices = 1.0 + 14.0 * tf.nn.sigmoid(num_voices_logit)  # [1, 16]

        detune_cents = params[:, 1:2] * 50.0  # [-50, 50] cents

        spread_amount = tf.nn.sigmoid(params[:, 2:3])  # [0, 1]

        phase_mod_depth = tf.nn.sigmoid(params[:, 3:4]) * 0.5  # [0, 0.5]

        # 時間変化するvoiceゲイン
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        voice_gains_latent = self.voice_gain_net(
            z_cond
        )  # [batch, latent_steps, max_voices]

        # 時間軸にアップサンプル
        voice_gains = tf.image.resize(
            tf.expand_dims(voice_gains_latent, axis=2),
            [TIME_LENGTH, 1],
            method="bilinear",
        )
        voice_gains = tf.squeeze(
            voice_gains, axis=2
        )  # [batch, time_length, max_voices]
        voice_gains = tf.nn.softplus(voice_gains)  # 正の値に

        # voiceごとのdetune適用
        unison_output = tf.zeros_like(base_signal)

        for voice_idx in range(self.max_voices):
            # voiceの位置（-0.5 ~ 0.5）
            voice_position = (
                voice_idx - (self.max_voices - 1) / 2.0
            ) / self.max_voices

            # detune量（cents）
            voice_detune_cents = voice_position * detune_cents[:, 0]

            # centsから周波数比に変換
            freq_ratio = tf.pow(2.0, voice_detune_cents / 1200.0)

            # 時間軸での位相シフト（デチューン近似）
            # より正確には周波数変調が必要だが、計算効率のため時間シフト
            phase_shift = voice_detune_cents * 0.1  # 経験的スケーリング
            shift_samples = tf.cast(phase_shift, tf.int32)

            # バッチごとのシフトは複雑なため、voiceインデックスベースのシフト
            shifted = tf.roll(
                base_signal,
                shift=int(voice_position * 20),  # -10 ~ 10サンプル
                axis=1,
            )

            # 位相変調（warmth）
            t = tf.cast(tf.range(TIME_LENGTH), tf.float32) / self.sr
            lfo_freq = 3.0 + voice_idx * 0.3
            phase_lfo = phase_mod_depth[:, 0:1] * tf.sin(
                2 * np.pi * lfo_freq * t
            )
            phase_lfo = phase_lfo[:, :TIME_LENGTH]

            modulated = shifted * (1.0 + phase_lfo * 0.2)

            # voiceゲインを適用
            voice_gain = voice_gains[:, :, voice_idx]

            # ステレオ的広がり（パンニング）
            pan_weight = (
                1.0 - tf.abs(voice_position) * spread_amount[:, 0:1] * 0.5
            )

            unison_output += modulated * voice_gain * pan_weight

        # ソフトマスキング: 学習されたnum_voicesに応じて出力を調整
        # num_voices=2なら少数voiceのみ、num_voices=16なら全voiceを使用
        voice_mask = tf.nn.sigmoid(
            (num_voices - tf.range(0.0, self.max_voices, dtype=tf.float32))
            * 2.0
        )
        voice_mask = tf.reshape(voice_mask, [batch_size, 1, self.max_voices])

        # 正規化（voice数に応じて自動調整）
        effective_voices = tf.reduce_sum(voice_mask, axis=-1, keepdims=True)
        normalization = tf.sqrt(effective_voices + 1e-6)

        unison_output_masked = (
            tf.reduce_sum(voice_gains * voice_mask, axis=-1)
            / (normalization[:, :, 0] + 1e-6)
            * unison_output
        )

        return unison_output_masked


# ========================================
# DDSP Module 2: Learnable Harmonic Distribution
# ========================================
class DDSPHarmonicDistribution(tf.keras.layers.Layer):
    """
    倍音分布を完全に学習:
    - 各倍音の振幅を時間変化として予測
    - 高周波強調もデータから学習
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # 倍音振幅の時間変化を予測
        self.harmonic_net = tf.keras.Sequential(
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
                    num_harmonics, 3, padding="same", activation=None
                ),
            ],
            name="harmonic_amps",
        )

        # 周波数依存の修正項（高周波強調など）
        self.freq_shaping_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_harmonics, activation=None),
            ],
            name="freq_shaping",
        )

    def call(self, z, cond, output_length=TIME_LENGTH):
        """
        Returns:
            amplitudes: [batch, time_length, num_harmonics]
        """
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        # 条件をブロードキャスト
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # 時間変化する倍音振幅
        amps_latent = self.harmonic_net(
            z_cond
        )  # [batch, latent_steps, num_harmonics]

        # アップサンプル
        amps = tf.image.resize(
            tf.expand_dims(amps_latent, axis=2),
            [output_length, 1],
            method="bilinear",
        )
        amps = tf.squeeze(amps, axis=2)  # [batch, time_length, num_harmonics]

        # 周波数依存のシェイピング
        z_global = tf.reduce_mean(z, axis=1)
        combined = tf.concat([z_global, cond], axis=-1)
        freq_shaping = self.freq_shaping_net(combined)  # [batch, num_harmonics]

        # 加算的に適用（学習の安定性向上）
        freq_shaping_expanded = freq_shaping[:, None, :]
        amps = amps + freq_shaping_expanded * 0.5

        # sigmoid で [0, 1] に正規化
        amps = tf.nn.sigmoid(amps)

        return amps


# ========================================
# DDSP Module 3: Learnable Envelope
# ========================================
class DDSPEnvelopeGenerator(tf.keras.layers.Layer):
    """
    エンベロープを完全に学習:
    - ADSR等の明示的構造なし
    - 時間変化する振幅を直接予測
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        self.envelope_net = tf.keras.Sequential(
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
                tf.keras.layers.Conv1D(1, 3, padding="same", activation=None),
            ],
            name="envelope",
        )

        # LFO的な変調を学習
        self.modulation_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(1, 3, padding="same", activation=None),
            ],
            name="modulation",
        )

    def call(self, z, cond):
        """
        Returns:
            envelope: [batch, time_length]
        """
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # ベースエンベロープ
        env_latent = self.envelope_net(z_cond)

        # アップサンプル
        env = tf.image.resize(
            tf.expand_dims(env_latent, axis=2),
            [self.output_length, 1],
            method="bilinear",
        )
        env = tf.squeeze(env, axis=[2, 3])

        # 変調成分
        mod_latent = self.modulation_net(z_cond)
        mod = tf.image.resize(
            tf.expand_dims(mod_latent, axis=2),
            [self.output_length, 1],
            method="bilinear",
        )
        mod = tf.squeeze(mod, axis=[2, 3])

        # 変調を加算的に適用
        final_envelope = env + mod * 0.3

        # sigmoid で [0, 1]
        final_envelope = tf.nn.sigmoid(final_envelope)

        return final_envelope


# ========================================
# DDSP Module 4: Learnable Noise
# ========================================
class DDSPNoiseGenerator(tf.keras.layers.Layer):
    """
    ノイズ成分を学習:
    - フィルタ特性
    - エンベロープ
    - 混合量
    """

    def __init__(self, output_length=TIME_LENGTH):
        super().__init__()
        self.output_length = output_length

        # ノイズエンベロープ
        self.noise_env_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(1, 3, padding="same", activation=None),
            ],
            name="noise_envelope",
        )

        # ノイズフィルタ
        self.noise_filter_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 15, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 11, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    16, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(1, 5, padding="same", activation="tanh"),
            ],
            name="noise_filter",
        )

        # 全体のノイズ量を予測
        self.noise_amount_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="noise_amount",
        )

    def call(self, z, cond):
        """
        Returns:
            noise: [batch, time_length]
        """
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # ノイズエンベロープ
        noise_env_latent = self.noise_env_net(z_cond)
        noise_env = tf.image.resize(
            tf.expand_dims(noise_env_latent, axis=2),
            [self.output_length, 1],
            method="bilinear",
        )
        noise_env = tf.squeeze(noise_env, axis=[2, 3])
        noise_env = tf.nn.sigmoid(noise_env)

        # ランダムノイズ生成
        random_noise = tf.random.normal([batch_size, self.output_length, 1])

        # フィルタリング
        filtered_noise = self.noise_filter_net(random_noise)
        filtered_noise = tf.squeeze(filtered_noise, axis=-1)

        # 全体のノイズ量
        z_global = tf.reduce_mean(z, axis=1)
        combined = tf.concat([z_global, cond], axis=-1)
        noise_amount = self.noise_amount_net(combined)
        noise_amount = noise_amount * 0.02  # スケーリング

        # 合成
        noise = filtered_noise * noise_env * noise_amount

        return noise


# ========================================
# DDSP Synthesizer: 全モジュールを統合
# ========================================
class DDSPSynthesizer(tf.keras.layers.Layer):
    """
    DDSPベースのシンセサイザー:
    倍音生成 → Unison → Envelope → Noise
    """

    def __init__(self, num_harmonics=NUM_HARMONICS, max_voices=16, sr=SR):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sr = float(sr)

        # DDSPモジュール
        self.harmonic_dist = DDSPHarmonicDistribution(num_harmonics)
        self.envelope_gen = DDSPEnvelopeGenerator()
        self.unison_detune = DDSPUnisonDetuneLayer(max_voices, sr)
        self.noise_gen = DDSPNoiseGenerator()

    def generate_harmonic_wave(self, fundamental_freq, amplitudes_time, phases):
        """
        倍音から波形生成（微分可能）
        """
        batch_size = tf.shape(fundamental_freq)[0]
        time_length = tf.shape(amplitudes_time)[1]
        num_harmonics = tf.shape(amplitudes_time)[2]

        t = tf.cast(tf.range(time_length), tf.float32) / self.sr
        t = tf.reshape(t, [1, -1, 1])
        f0 = tf.reshape(fundamental_freq, [-1, 1, 1])
        harmonic_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
        harmonic_nums = tf.reshape(harmonic_nums, [1, 1, -1])

        omega = 2.0 * np.pi * f0 * harmonic_nums
        phas = tf.reshape(phases, [-1, 1, num_harmonics])

        harmonics = amplitudes_time * tf.sin(omega * t + phas)
        wave = tf.reduce_sum(harmonics, axis=-1)

        return wave

    def call(self, z, cond):
        """
        Args:
            z: [batch, latent_steps, latent_dim]
            cond: [batch, cond_dim]  # [pitch, timbre1, timbre2, timbre3]

        Returns:
            output: [batch, time_length, 1]
        """
        batch_size = tf.shape(z)[0]

        # 1. ピッチから基本周波数
        pitch = cond[:, 0]
        fundamental_freq = 440.0 * tf.pow(
            2.0, ((pitch * 35.0 + 36.0) - 69.0) / 12.0
        )

        # 2. 倍音分布を予測
        harmonic_amps = self.harmonic_dist(
            z, cond
        )  # [batch, time, num_harmonics]

        # 3. 位相初期化（ゼロ位相）
        initial_amps = harmonic_amps[:, 0, :]
        phases = tf.zeros_like(initial_amps)

        # 4. 倍音から基本波形生成
        base_harmonic_wave = self.generate_harmonic_wave(
            fundamental_freq, harmonic_amps, phases
        )

        # 5. Unison/Detune適用
        thick_wave = self.unison_detune(
            base_harmonic_wave, z, cond, fundamental_freq
        )

        # 6. エンベロープ適用
        envelope = self.envelope_gen(z, cond)
        enveloped_wave = thick_wave * envelope

        # 7. ノイズ追加
        noise = self.noise_gen(z, cond)

        # 8. 最終合成
        output = enveloped_wave + noise

        # tanh で [-1, 1] にクリップ
        output = tf.nn.tanh(output)

        # [batch, time, 1]
        output = tf.expand_dims(output, axis=-1)

        return output


# ========================================
# Encoder (変更なし)
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
# DDSP Decoder
# ========================================
def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # DDSPシンセサイザー
    synth = DDSPSynthesizer(num_harmonics=NUM_HARMONICS, max_voices=16, sr=SR)

    output = synth(z_in, cond)

    return tf.keras.Model([z_in, cond], output, name="ddsp_decoder")


# ========================================
# TimeWiseCVAE with DDSP
# ========================================
class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
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
        """推論用: ゼロベクトルから生成"""
        z = tf.zeros((tf.shape(cond)[0], LATENT_STEPS, LATENT_DIM))
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

            loss = (
                recon * 25.0
                + stft_loss * 12.0
                + mel_loss * 10.0
                + kl_divergence * kl_weight
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
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
