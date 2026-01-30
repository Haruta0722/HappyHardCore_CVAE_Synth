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
# 高周波・スペクトルフラックス損失
# ========================================
def compute_spectral_flux_loss(
    y_true, y_pred, frame_length=2048, hop_length=512
):
    """
    スペクトルフラックスの差を損失として計算
    screech特有の時間変化を学習
    """

    def get_spectral_flux(audio):
        # STFT
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=hop_length,
            fft_length=frame_length,
        )
        magnitude = tf.abs(stft)

        # フレーム間の差分
        diff = magnitude[:, 1:, :] - magnitude[:, :-1, :]
        # 正の変化のみ考慮（増加のみ）
        diff = tf.maximum(diff, 0.0)

        # 時間軸で平均
        flux = tf.reduce_mean(diff, axis=-1)  # [batch, frames-1]
        return flux

    flux_true = get_spectral_flux(y_true)
    flux_pred = get_spectral_flux(y_pred)

    # L1距離
    loss = tf.reduce_mean(tf.abs(flux_true - flux_pred))
    return loss


def compute_high_freq_emphasis_loss(y_true, y_pred, sr=SR, cutoff_freq=2000.0):
    """
    高周波帯域のエネルギー差を重視した損失
    screechの高周波成分を強化
    """
    # STFT
    stft_true = tf.signal.stft(
        y_true, frame_length=2048, frame_step=512, fft_length=2048
    )
    stft_pred = tf.signal.stft(
        y_pred, frame_length=2048, frame_step=512, fft_length=2048
    )

    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)

    # 周波数ビンのインデックス
    num_bins = tf.shape(mag_true)[-1]
    freq_bins = (
        tf.cast(tf.range(num_bins), tf.float32)
        * (sr / 2.0)
        / tf.cast(num_bins, tf.float32)
    )

    # 高周波の重み（2kHz以上を強調）
    high_freq_weight = tf.sigmoid(
        (freq_bins - cutoff_freq) / 500.0
    )  # [num_bins]
    high_freq_weight = tf.reshape(high_freq_weight, [1, 1, -1])

    # 重み付き損失
    weighted_diff = tf.square(mag_true - mag_pred) * (
        1.0 + high_freq_weight * 5.0
    )
    loss = tf.reduce_mean(weighted_diff)

    return loss


# ========================================
# ★改良版: 適応的ノイズジェネレータ（倍音構造を保護）
# ========================================
class AdaptiveNoiseGenerator(tf.keras.layers.Layer):
    """
    周波数帯域別にノイズを生成
    音色ごとに最適なノイズ量を自動調整（pluckは少なく）
    ★倍音構造を破壊しないように、ノイズを倍音の「隙間」に配置
    """

    def __init__(self, output_length=TIME_LENGTH, num_bands=4):
        super().__init__()
        self.output_length = output_length
        self.num_bands = num_bands

        # 各周波数帯域のノイズエンベロープ
        self.band_envelope_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    128, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    64, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    num_bands, 3, padding="same", activation="sigmoid"
                ),
            ],
            name="band_envelopes",
        )

        # 各帯域のフィルタ特性
        self.band_filters = []
        for i in range(num_bands):
            self.band_filters.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv1D(
                            16, 15, padding="same", activation="relu"
                        ),
                        tf.keras.layers.Conv1D(
                            8, 9, padding="same", activation="relu"
                        ),
                        tf.keras.layers.Conv1D(
                            1, 5, padding="same", activation="tanh"
                        ),
                    ],
                    name=f"band_filter_{i}",
                )
            )

        # ★全体のノイズ量を学習（音色から）- より控えめに
        self.noise_intensity_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(
                    num_bands + 1, activation="sigmoid"
                ),  # +1は全体ゲイン
            ],
            name="noise_intensity",
        )

    def call(self, z, cond):
        batch_size = tf.shape(z)[0]
        latent_steps = tf.shape(z)[1]

        # 条件をブロードキャスト
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        # 各帯域のエンベロープ
        band_envelopes = self.band_envelope_net(
            z_cond
        )  # [batch, latent_steps, num_bands]
        band_envelopes = tf.image.resize(
            tf.expand_dims(band_envelopes, axis=2),
            [self.output_length, 1],
            method="bilinear",
        )
        band_envelopes = tf.squeeze(
            band_envelopes, axis=2
        )  # [batch, time, num_bands]

        # ★NaN防止: クリッピング
        band_envelopes = tf.clip_by_value(band_envelopes, 0.0, 1.0)

        # 音色から各帯域のノイズ強度 + 全体ゲイン
        timbre = cond[:, 1:]
        noise_params = self.noise_intensity_net(timbre)  # [batch, num_bands+1]

        # ★NaN防止: クリッピング
        noise_params = tf.clip_by_value(noise_params, 0.0, 1.0)

        noise_intensities = noise_params[:, : self.num_bands]  # 各帯域
        global_noise_gain = noise_params[
            :, self.num_bands : self.num_bands + 1
        ]  # 全体ゲイン

        # 各帯域のノイズを生成・合成
        total_noise = tf.zeros([batch_size, self.output_length])

        for i in range(self.num_bands):
            # ホワイトノイズ
            raw_noise = tf.random.normal([batch_size, self.output_length, 1])

            # 帯域フィルタリング
            filtered = self.band_filters[i](raw_noise)
            filtered = tf.squeeze(filtered, axis=-1)

            # ★NaN防止: クリッピング
            filtered = tf.clip_by_value(filtered, -10.0, 10.0)

            # エンベロープと強度を適用
            band_env = band_envelopes[:, :, i]
            intensity = noise_intensities[:, i : i + 1]

            # ★高周波帯域ほど強くするが、過度にならないように調整
            freq_boost = 1.0 + (i / float(self.num_bands)) * 1.2  # 2.0 → 1.2

            band_noise = filtered * band_env * intensity * freq_boost
            total_noise = total_noise + band_noise

        # 正規化と全体ゲイン適用
        total_noise = total_noise / float(self.num_bands)
        # ★全体的なノイズレベルを削減（1.5 → 0.8）
        total_noise = total_noise * global_noise_gain * 0.5  # 0.8 → 0.5

        # ★NaN防止: 最終クリッピング
        total_noise = tf.clip_by_value(total_noise, -1.0, 1.0)

        return total_noise


# ========================================
# ★新規: 奇数倍音強調レイヤー
# ========================================
class OddHarmonicEnhancer(tf.keras.layers.Layer):
    """
    奇数倍音を強調し、偶数倍音を抑制
    音色によって強調度を調整
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # 音色ごとの奇数倍音強調度を学習
        self.odd_enhancement_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(
                    1, activation="sigmoid"
                ),  # 奇数倍音強調度 [0, 1]
            ],
            name="odd_enhancement",
        )

    def call(self, amplitudes, timbre):
        """
        Args:
            amplitudes: [batch, time_length, num_harmonics]
            timbre: [batch, 3]

        Returns:
            enhanced_amplitudes: [batch, time_length, num_harmonics]
        """
        # 音色から奇数倍音強調度を取得
        odd_strength = self.odd_enhancement_net(timbre)  # [batch, 1]

        # ★NaN防止: クリッピング
        odd_strength = tf.clip_by_value(odd_strength, 0.0, 1.0)

        # 倍音インデックス (1, 2, 3, ...)
        harmonic_indices = tf.range(1, self.num_harmonics + 1, dtype=tf.float32)

        # 奇数倍音マスク (1, 3, 5, ...) → 1.0、偶数倍音 (2, 4, 6, ...) → 0.0
        is_odd = tf.cast(tf.math.mod(harmonic_indices, 2.0) > 0.5, tf.float32)

        # 奇数倍音の強調プロファイル
        # odd_strength=0の場合: すべて1.0（変化なし）
        # odd_strength=1の場合: 奇数=2.0、偶数=0.5
        odd_profile = is_odd * (1.0 + odd_strength[:, :, None]) + (
            1.0 - is_odd
        ) * (1.0 - 0.5 * odd_strength[:, :, None])

        # ★NaN防止: プロファイルを安全な範囲に制限
        odd_profile = tf.clip_by_value(odd_profile, 0.3, 2.5)

        odd_profile = tf.reshape(odd_profile, [-1, 1, self.num_harmonics])

        # 適用
        enhanced_amps = amplitudes * odd_profile

        # ★NaN防止: 出力を制限
        enhanced_amps = tf.clip_by_value(enhanced_amps, 0.0, 2.0)

        return enhanced_amps


# ========================================
# 周波数帯域別倍音コントローラ
# ========================================
class FrequencyBandHarmonicController(tf.keras.layers.Layer):
    """
    低・中・高周波帯域で異なる倍音制御
    screechは高周波倍音が強い特性を学習
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__()
        self.num_harmonics = num_harmonics

        # 低・中・高周波帯域の制御パラメータ
        self.band_control_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(
                    3, activation="sigmoid"
                ),  # [low_gain, mid_gain, high_gain]
            ],
            name="band_control",
        )

    def call(self, amplitudes, timbre, z_features):
        """
        Args:
            amplitudes: [batch, time_length, num_harmonics]
            timbre: [batch, 3]
            z_features: [batch, feature_dim]
        """
        # 音色とzから帯域ゲインを計算
        combined = tf.concat([z_features, timbre], axis=-1)
        band_gains = self.band_control_net(combined)  # [batch, 3]

        # ★NaN防止: クリッピング
        band_gains = tf.clip_by_value(band_gains, 0.0, 1.0)

        low_gain = band_gains[:, 0:1]
        mid_gain = band_gains[:, 1:2]
        high_gain = band_gains[:, 2:3]

        # 倍音インデックス
        harmonic_indices = tf.cast(
            tf.range(1, self.num_harmonics + 1), tf.float32
        )

        # 各倍音を周波数帯域に分類
        # 低域: 1-8, 中域: 9-20, 高域: 21-32
        low_mask = tf.cast(harmonic_indices <= 8, tf.float32)
        mid_mask = tf.cast(
            (harmonic_indices > 8) & (harmonic_indices <= 20), tf.float32
        )
        high_mask = tf.cast(harmonic_indices > 20, tf.float32)

        # ★各帯域のゲインを適用（高域の過度な強調を抑制 2.0 → 1.5）
        gain_profile = (
            low_mask * low_gain[:, :, None]
            + mid_mask * mid_gain[:, :, None]
            + high_mask * high_gain[:, :, None] * 1.5
        )

        gain_profile = tf.reshape(gain_profile, [-1, 1, self.num_harmonics])

        # ★NaN防止: ゲインプロファイルを制限
        gain_profile = tf.clip_by_value(gain_profile, 0.0, 1.5)

        # 適用
        controlled_amps = amplitudes * (0.5 + gain_profile)

        # ★NaN防止: 出力を制限
        controlled_amps = tf.clip_by_value(controlled_amps, 0.0, 2.0)

        return controlled_amps


# ========================================
# ★改良版HarmonicAmplitudeNet（奇数倍音強調を追加）
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

        # zから特徴を抽出
        self.z_feature_extractor = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation="relu"),
            ],
            name="z_features",
        )

        # 周波数帯域別制御
        self.band_controller = FrequencyBandHarmonicController(num_harmonics)

        # ★奇数倍音強調
        self.odd_enhancer = OddHarmonicEnhancer(num_harmonics)

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

        # zから特徴を抽出
        z_features = self.z_feature_extractor(z)

        # 周波数帯域別制御
        timbre = cond[:, 1:]
        amps = self.band_controller(amps, timbre, z_features)

        # ★奇数倍音強調
        amps = self.odd_enhancer(amps, timbre)

        return amps


# ========================================
# ThicknessGenerator: 音の厚みを生成
# ========================================
class ThicknessGenerator(tf.keras.layers.Layer):
    """
    音の厚みを実現:
    1. 複数ボイス（Unison/Chorus効果）
    2. 微細な周波数変動（Detune）
    3. 位相変調（Warmth）
    """

    def __init__(self, num_voices=3, sr=SR):
        super().__init__()
        self.num_voices = num_voices
        self.sr = sr

        # 音色ごとの厚みパラメータ
        self.thickness_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(
                    3, activation="sigmoid"
                ),  # [detune, spread, phase_mod]
            ],
            name="thickness_params",
        )

        # zからの時間変化する厚みエンベロープ
        self.voice_envelope_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, 7, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    32, 5, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv1D(
                    num_voices, 3, padding="same", activation="sigmoid"
                ),
            ],
            name="voice_envelopes",
        )

    def call(self, base_harmonics, z, cond, fundamental_freq):
        """
        Args:
            base_harmonics: 基本倍音波形 [batch, time_length]
            z: 潜在変数 [batch, latent_steps, latent_dim]
            cond: 条件 [batch, 4]
            fundamental_freq: 基本周波数 [batch]

        Returns:
            thick_harmonics: 厚みのある倍音波形 [batch, time_length]
        """
        batch_size = tf.shape(base_harmonics)[0]
        time_length = tf.shape(base_harmonics)[1]

        # 音色から厚みパラメータを取得
        timbre = cond[:, 1:]
        thickness_params = self.thickness_param_net(timbre)

        # ★NaN防止: クリッピング
        thickness_params = tf.clip_by_value(thickness_params, 0.0, 1.0)

        detune_amount = thickness_params[:, 0:1] * 0.02  # 最大2%
        spread_amount = thickness_params[:, 1:2] * 0.5
        phase_mod_depth = thickness_params[:, 2:3] * 0.3

        # zから時間変化する各ボイスのエンベロープ
        latent_steps = tf.shape(z)[1]
        cond_broadcast = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        z_cond = tf.concat([z, cond_broadcast], axis=-1)

        voice_envelopes = self.voice_envelope_net(
            z_cond
        )  # [batch, latent_steps, num_voices]

        # 時間軸にアップサンプル
        voice_envelopes = tf.image.resize(
            tf.expand_dims(voice_envelopes, axis=2),
            [TIME_LENGTH, 1],
            method="bilinear",
        )
        voice_envelopes = tf.squeeze(
            voice_envelopes, axis=2
        )  # [batch, time_length, num_voices]

        # ★NaN防止: クリッピング
        voice_envelopes = tf.clip_by_value(voice_envelopes, 0.0, 1.0)

        # 複数ボイスを生成
        thick_output = tf.zeros_like(base_harmonics)

        for voice_idx in range(self.num_voices):
            # 各ボイスのデチューン量
            detune_offset = (
                voice_idx - (self.num_voices - 1) / 2
            ) / self.num_voices
            detune_factor = detune_offset * detune_amount[:, 0]

            # 位相シフトで擬似的にデチューン効果
            # 実際の周波数変調の代わりに時間シフトで近似
            shift_samples = tf.cast(detune_factor * 10.0, tf.int32)

            # 各バッチアイテムごとにシフト
            shifted_harmonics = tf.roll(
                base_harmonics, shift=voice_idx - 1, axis=1
            )

            # 位相変調（LFOによる揺らぎ）
            t = tf.cast(tf.range(TIME_LENGTH), tf.float32) / self.sr
            lfo_rate = 5.0 + voice_idx * 2.0
            phase_lfo = phase_mod_depth[:, 0:1] * tf.sin(
                2 * np.pi * lfo_rate * t
            )
            phase_lfo = phase_lfo[:, :TIME_LENGTH]

            # ★NaN防止: クリッピング
            phase_lfo = tf.clip_by_value(phase_lfo, -1.0, 1.0)

            # 簡易的な位相変調（振幅変調で近似）
            modulated = shifted_harmonics * (1.0 + phase_lfo * 0.1)

            # ★NaN防止: クリッピング
            modulated = tf.clip_by_value(modulated, -10.0, 10.0)

            # ボイスエンベロープを適用
            voice_env = voice_envelopes[:, :, voice_idx]

            # パンニング（ステレオ的広がり）
            pan = (voice_idx - (self.num_voices - 1) / 2) / self.num_voices
            pan_weight = 1.0 - tf.abs(pan) * spread_amount[:, 0:1]

            # ★NaN防止: クリッピング
            pan_weight = tf.clip_by_value(pan_weight, 0.0, 1.0)

            thick_output += modulated * voice_env * pan_weight

        # 正規化
        thick_output = thick_output / float(self.num_voices)

        # ★NaN防止: 最終クリッピング
        thick_output = tf.clip_by_value(thick_output, -10.0, 10.0)

        return thick_output


# ========================================
# 改良版TimbreEnvelopeShaper（音色別の柔軟性向上）
# ========================================
class TimbreEnvelopeShaper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.adsr_param_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(4, activation="sigmoid"),  # 4パラメータ
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

        # ★NaN防止: クリッピング
        adsr_params = tf.clip_by_value(adsr_params, 0.0, 1.0)

        attack_speed = adsr_params[:, 0:1] * 25.0 + 1.0
        decay_rate = adsr_params[:, 1:2] * 20.0 + 0.5
        sustain_level = adsr_params[:, 2:3] * 0.9 + 0.05
        sustain_duration = adsr_params[:, 3:4] * 0.5 + 0.2

        lfo_params = self.lfo_param_net(combined_input)

        # ★NaN防止: クリッピング
        lfo_params = tf.clip_by_value(lfo_params, 0.0, 1.0)

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
        # ★NaN防止: ゼロ除算対策とクリッピング
        attack_curve = tf.pow(
            tf.clip_by_value(t / (attack_ratio + 1e-6), 0.0, 1.0),
            1.0 / tf.clip_by_value(attack_speed, 0.1, 50.0),
        )

        decay_end = attack_ratio + decay_ratio
        decay_mask = tf.cast((t >= attack_ratio) & (t < decay_end), tf.float32)
        decay_t = (t - attack_ratio) / (decay_ratio + 1e-6)
        decay_curve = 1.0 - (1.0 - sustain_level) * (
            1.0 - tf.exp(-tf.clip_by_value(decay_rate, 0.1, 50.0) * decay_t)
        )

        sustain_end = decay_end + sustain_duration[:, 0:1]
        sustain_mask = tf.cast((t >= decay_end) & (t < sustain_end), tf.float32)
        sustain_curve = sustain_level

        release_mask = tf.cast(t >= sustain_end, tf.float32)
        release_t = (t - sustain_end) / (1.0 - sustain_end + 1e-6)
        release_curve = sustain_level * tf.exp(
            -tf.clip_by_value(decay_rate, 0.1, 50.0) * release_t
        )

        envelope_shape = (
            attack_mask * attack_curve
            + decay_mask * decay_curve
            + sustain_mask * sustain_curve
            + release_mask * release_curve
        )

        # ★NaN防止: クリッピング
        envelope_shape = tf.clip_by_value(envelope_shape, 0.0, 1.0)

        t_seconds = t * WAV_LENGTH
        lfo_modulation = 1.0 + lfo_depth * tf.sin(
            2.0 * np.pi * lfo_rate * t_seconds
        )

        # ★NaN防止: クリッピング
        lfo_modulation = tf.clip_by_value(lfo_modulation, 0.3, 1.7)

        final_envelope = base_envelope * 0.3 + envelope_shape * 0.7
        final_envelope = final_envelope * lfo_modulation

        # ★NaN防止: 最終クリッピング
        final_envelope = tf.clip_by_value(final_envelope, 0.0, 1.5)

        return final_envelope


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
# ★改良版Decoder: ノイズを控えめに、倍音構造を保護
# ========================================
def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    # 倍音振幅生成（周波数帯域別制御 + 奇数倍音強調）
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

    # ★適応的ノイズ生成（倍音構造を保護する控えめな設定）
    adaptive_noise_gen = AdaptiveNoiseGenerator(num_bands=4)
    noise = adaptive_noise_gen(z_in, cond)

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

    # 基本倍音波形生成
    harmonic_wave_layer = GenerateHarmonicWaveTimeVarying()
    base_harmonic_wave = harmonic_wave_layer(
        [fundamental_freq, shaped_harmonic_amps, phases]
    )

    # 厚みを追加
    thickness_gen = ThicknessGenerator(num_voices=3)
    thick_harmonic_wave = thickness_gen(
        base_harmonic_wave, z_in, cond, fundamental_freq
    )

    # ★最終合成: 倍音を優先し、ノイズは控えめに混ぜる
    # 倍音:ノイズ = 約 10:1 の比率
    harmonic_component = thick_harmonic_wave * envelope

    # ★NaN防止: クリッピング（Lambdaレイヤーでラップ）
    harmonic_component = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -10.0, 10.0)
    )(harmonic_component)

    noise_component = noise * 0.1  # ノイズを10%に抑制

    # ★NaN防止: クリッピング（Lambdaレイヤーでラップ）
    noise_component = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -1.0, 1.0)
    )(noise_component)

    output = harmonic_component + noise_component

    # ★NaN防止: tanh前にクリッピング（Lambdaレイヤーでラップ）
    output = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -10.0, 10.0))(
        output
    )

    output = tf.keras.layers.Activation("tanh")(output)
    output = tf.keras.layers.Lambda(lambda x: x[:, :, None])(output)

    return tf.keras.Model([z_in, cond], output, name="decoder")


# ========================================
# TimeWiseCVAE: 高周波・スペクトルフラックス損失を追加
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

            # ★NaN防止: z_logvarをクリッピング
            z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)

            z = sample_z(z_mean, z_logvar)

            # ★NaN防止: zをクリッピング
            z = tf.clip_by_value(z, -10.0, 10.0)

            x_hat = self.decoder([z, cond], training=True)
            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            # ★NaN防止: 入力と出力をクリッピング
            x_target = tf.clip_by_value(x_target, -1.0, 1.0)
            x_hat_sq = tf.clip_by_value(x_hat_sq, -1.0, 1.0)

            # 基本的な再構成損失
            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))

            # ★NaN防止: 再構成損失をチェック
            recon = tf.where(tf.math.is_nan(recon), 1.0, recon)
            recon = tf.where(tf.math.is_inf(recon), 1.0, recon)

            # 既存のスペクトル損失
            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★NaN防止: スペクトル損失をチェック
            stft_loss = tf.where(tf.math.is_nan(stft_loss), 1.0, stft_loss)
            stft_loss = tf.where(tf.math.is_inf(stft_loss), 1.0, stft_loss)
            mel_loss = tf.where(tf.math.is_nan(mel_loss), 1.0, mel_loss)
            mel_loss = tf.where(tf.math.is_inf(mel_loss), 1.0, mel_loss)

            # 高周波強調損失（screechの高周波特性のため）
            high_freq_loss = compute_high_freq_emphasis_loss(x_target, x_hat_sq)

            # ★NaN防止
            high_freq_loss = tf.where(
                tf.math.is_nan(high_freq_loss), 1.0, high_freq_loss
            )
            high_freq_loss = tf.where(
                tf.math.is_inf(high_freq_loss), 1.0, high_freq_loss
            )

            # スペクトルフラックス損失
            spectral_flux_loss = compute_spectral_flux_loss(x_target, x_hat_sq)

            # ★NaN防止
            spectral_flux_loss = tf.where(
                tf.math.is_nan(spectral_flux_loss), 0.1, spectral_flux_loss
            )
            spectral_flux_loss = tf.where(
                tf.math.is_inf(spectral_flux_loss), 0.1, spectral_flux_loss
            )

            # KL損失
            kl_per_dim = -0.5 * (
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            kl_divergence = tf.reduce_mean(
                tf.maximum(kl_per_dim, self.free_bits)
            )

            # ★NaN防止
            kl_divergence = tf.where(
                tf.math.is_nan(kl_divergence), 0.5, kl_divergence
            )
            kl_divergence = tf.where(
                tf.math.is_inf(kl_divergence), 0.5, kl_divergence
            )
            kl_divergence = tf.clip_by_value(kl_divergence, 0.0, 10.0)

            kl_weight = self.compute_kl_weight()

            # ★総合損失: 高周波損失を削減、メル損失を増加（倍音構造保護のため）
            loss = (
                recon * 25.0
                + stft_loss * 12.0
                + mel_loss * 15.0  # 10.0 → 15.0
                + high_freq_loss * 8.0  # 15.0 → 8.0
                + spectral_flux_loss * 8.0
                + kl_divergence * kl_weight
            )

            # ★NaN防止: 総合損失をチェック
            loss = tf.where(tf.math.is_nan(loss), 1000.0, loss)
            loss = tf.where(tf.math.is_inf(loss), 1000.0, loss)

        grads = tape.gradient(loss, self.trainable_variables)

        # ★NaN防止: 勾配をチェック
        grads = [
            (
                tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
                if g is not None
                else None
            )
            for g in grads
        ]
        grads = [
            (
                tf.where(tf.math.is_inf(g), tf.zeros_like(g), g)
                if g is not None
                else None
            )
            for g in grads
        ]

        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)

        # ★NaN防止: grad_normをチェック
        grad_norm = tf.where(tf.math.is_nan(grad_norm), 0.0, grad_norm)
        grad_norm = tf.where(tf.math.is_inf(grad_norm), 0.0, grad_norm)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))

        # ★NaN防止
        z_std = tf.where(tf.math.is_nan(z_std), 1.0, z_std)
        z_std = tf.where(tf.math.is_inf(z_std), 1.0, z_std)

        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)
        self.best_recon.assign(tf.minimum(self.best_recon, recon))

        return {
            "loss": loss,
            "recon": recon,
            "best_recon": self.best_recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "high_freq": high_freq_loss,
            "spectral_flux": spectral_flux_loss,
            "kl": kl_divergence,
            "kl_weight": kl_weight,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
