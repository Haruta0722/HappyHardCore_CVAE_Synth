"""
dsp.py  ―  DDSPシンセサイザーモジュール

【役割】
  VAE (cvae.py) が出力した DDSPパラメータを受け取り、音声波形を合成する。
  学習パラメータを一切持たない純粋なDSP処理のみで構成されており、
  TensorFlow に依存しない NumPy 版も提供する。

【ハード化時の使い方】
  DSP専用マイコンはこのファイルだけを参照すればよい。
  VAE側から受け取るペイロード (DDSPParams) の構造:

    {
      "amplitudes":             float32 [LATENT_STEPS, 1],
      "harmonic_distribution":  float32 [LATENT_STEPS, NUM_HARMONICS],
      "noise_magnitudes":       float32 [LATENT_STEPS, NOISE_FIR_LEN],
      "f0_hz":                  float32  (スカラー)
    }

  受け取ったペイロードを NumPy版 synthesize_numpy() に渡すだけで音声が得られる。

【モジュール構成】
  upsample_frames()          フレーム→サンプルの補間 (TF版)
  HarmonicSynthesizer        加算合成シンセ (TFレイヤー, 学習時用)
  FilteredNoiseSynthesizer   ノイズシンセ   (TFレイヤー, 学習時用)
  synthesize_numpy()         NumPy実装 (マイコン・推論専用)
"""

import numpy as np

# ── TensorFlow は任意依存 (NumPy版のみ使う場合は不要) ───────────────────
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

from config import SR, TIME_LENGTH, NUM_HARMONICS, NOISE_FIR_LEN


# ============================================================
# ユーティリティ: フレーム → サンプルのアップサンプル (TF版)
# ============================================================
def upsample_frames(x, target_length):
    """
    [batch, frames, ch] → [batch, target_length, ch]  bilinear補間 (TF版)

    HarmonicSynthesizer / FilteredNoiseSynthesizer の内部で使用。
    """
    x = tf.expand_dims(x, axis=2)
    x = tf.image.resize(x, [target_length, 1], method="bilinear")
    x = tf.squeeze(x, axis=2)
    return x


def upsample_frames_numpy(x, target_length):
    """
    [frames, ch] → [target_length, ch]  線形補間 (NumPy版)

    synthesize_numpy() の内部で使用。マイコンでも動作する。
    """
    frames, ch = x.shape
    result = np.zeros((target_length, ch), dtype=np.float32)
    for c in range(ch):
        result[:, c] = np.interp(
            np.linspace(0, frames - 1, target_length),
            np.arange(frames),
            x[:, c],
        )
    return result


# ============================================================
# HarmonicSynthesizer  (TFレイヤー版 / 学習・推論用)
# ============================================================
class HarmonicSynthesizer(tf.keras.layers.Layer if HAS_TF else object):
    """
    加算合成シンセサイザー (DDSP論文 Section 3.1)

    cumsum による累積位相計算でフレーム境界の位相連続性を保証。

    入力:
      f0_hz                : 基本周波数 [batch]
      amplitudes           : 全体振幅   [batch, frames, 1]
      harmonic_distribution: 各倍音の相対比率 [batch, frames, NUM_HARMONICS]
    出力:
      audio: [batch, TIME_LENGTH]
    """

    def __init__(self, sr=SR, time_length=TIME_LENGTH):
        super().__init__(name="harmonic_synth")
        self.sr          = float(sr)
        self.time_length = time_length

    def call(self, f0_hz, amplitudes, harmonic_distribution):
        num_harmonics = tf.shape(harmonic_distribution)[2]
        time_frames   = tf.shape(amplitudes)[1]

        # 各倍音の絶対振幅  [batch, frames, num_harmonics]
        harm_amps = amplitudes * harmonic_distribution

        # f0 → 各倍音周波数  [batch, frames, num_harmonics]
        f0_frames  = tf.reshape(f0_hz, [-1, 1, 1])
        f0_frames  = tf.tile(f0_frames, [1, time_frames, 1])
        harm_nums  = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
        harm_freqs = f0_frames * tf.reshape(harm_nums, [1, 1, -1])

        # フレーム → サンプルにアップサンプル
        harm_amps  = upsample_frames(harm_amps,  self.time_length)
        harm_freqs = upsample_frames(harm_freqs, self.time_length)

        harm_amps  = tf.clip_by_value(harm_amps,  0.0, 2.0)
        harm_freqs = tf.clip_by_value(harm_freqs, 0.0, self.sr / 2.0)

        # 累積位相 → sin 波
        delta_phase = 2.0 * np.pi * harm_freqs / self.sr
        phase       = tf.cumsum(delta_phase, axis=1)
        harmonics   = harm_amps * tf.sin(phase)
        audio       = tf.reduce_sum(harmonics, axis=-1)
        return tf.clip_by_value(audio, -10.0, 10.0)


# ============================================================
# FilteredNoiseSynthesizer  (TFレイヤー版 / 学習・推論用)
# ============================================================
class FilteredNoiseSynthesizer(tf.keras.layers.Layer if HAS_TF else object):
    """
    ノイズシンセサイザー (DDSP論文 Section 3.2)

    ホワイトノイズを noise_magnitudes のフレームごとの平均振幅で整形する。

    入力:
      noise_magnitudes: [batch, frames, NOISE_FIR_LEN]
    出力:
      audio: [batch, TIME_LENGTH]
    """

    def __init__(self, sr=SR, time_length=TIME_LENGTH):
        super().__init__(name="noise_synth")
        self.sr          = sr
        self.time_length = time_length

    def call(self, noise_magnitudes):
        batch_size = tf.shape(noise_magnitudes)[0]
        noise      = tf.random.normal([batch_size, self.time_length])

        envelope = tf.reduce_mean(noise_magnitudes, axis=-1, keepdims=True)
        envelope = upsample_frames(envelope, self.time_length)
        envelope = tf.squeeze(envelope, axis=-1)
        envelope = tf.clip_by_value(envelope, 0.0, 1.0)

        return tf.clip_by_value(noise * envelope, -2.0, 2.0)


# ============================================================
# synthesize_numpy()  ― NumPy実装 (マイコン・推論専用)
# ============================================================
def synthesize_numpy(
    f0_hz:                 float,
    amplitudes:            np.ndarray,   # [LATENT_STEPS, 1]
    harmonic_distribution: np.ndarray,   # [LATENT_STEPS, NUM_HARMONICS]
    noise_magnitudes:      np.ndarray,   # [LATENT_STEPS, NOISE_FIR_LEN]
    sr:          int   = SR,
    time_length: int   = TIME_LENGTH,
    noise_gain:  float = 0.1,
    seed:        int   = None,
) -> np.ndarray:
    """
    DDSPパラメータから音声波形を合成する (NumPy実装)。

    TensorFlow 不要。マイコンや軽量環境で動作する。

    Args:
        f0_hz                : 基本周波数 [Hz]  (スカラー)
        amplitudes           : 全体振幅エンベロープ [LATENT_STEPS, 1]
        harmonic_distribution: 各倍音の相対比率   [LATENT_STEPS, NUM_HARMONICS]
        noise_magnitudes     : ノイズFIRフィルタ係数 [LATENT_STEPS, NOISE_FIR_LEN]
        sr                   : サンプリングレート
        time_length          : 出力サンプル数
        noise_gain           : ノイズの混合比率 (デフォルト 0.1)
        seed                 : 乱数シード (再現性が必要な場合に設定)

    Returns:
        audio: float32 ndarray [time_length]  (tanh 適用済み, -1〜1)

    使用例 (DSP マイコン側):
        import numpy as np
        from dsp import synthesize_numpy

        payload = receive_from_vae_mcu()   # 通信で受け取るペイロード
        audio = synthesize_numpy(
            f0_hz                 = payload["f0_hz"],
            amplitudes            = np.array(payload["amplitudes"]),
            harmonic_distribution = np.array(payload["harmonic_distribution"]),
            noise_magnitudes      = np.array(payload["noise_magnitudes"]),
        )
        play_audio(audio)
    """
    if seed is not None:
        np.random.seed(seed)

    num_harmonics = harmonic_distribution.shape[1]

    # ── 加算合成 ──────────────────────────────────────────────────────
    # 各倍音の絶対振幅  [LATENT_STEPS, num_harmonics]
    harm_amps = amplitudes * harmonic_distribution   # broadcast: [steps,1] * [steps,H]

    # フレーム → サンプルにアップサンプル
    harm_amps_up = upsample_frames_numpy(harm_amps, time_length)   # [time, H]

    # 各倍音の周波数  [LATENT_STEPS, num_harmonics]
    harm_nums  = np.arange(1, num_harmonics + 1, dtype=np.float32)
    harm_freqs = np.full((harmonic_distribution.shape[0], num_harmonics),
                         f0_hz, dtype=np.float32) * harm_nums[None, :]

    harm_freqs_up = upsample_frames_numpy(harm_freqs, time_length)  # [time, H]
    harm_freqs_up = np.clip(harm_freqs_up, 0.0, sr / 2.0)
    harm_amps_up  = np.clip(harm_amps_up, 0.0, 2.0)

    # 累積位相で sin 波を生成 (位相連続性保証)
    delta_phase   = 2.0 * np.pi * harm_freqs_up / sr   # [time, H]
    phase         = np.cumsum(delta_phase, axis=0)      # [time, H]
    harmonics     = harm_amps_up * np.sin(phase)        # [time, H]
    audio_harmonic = harmonics.sum(axis=1)              # [time]

    # ── ノイズ合成 ────────────────────────────────────────────────────
    noise_env = noise_magnitudes.mean(axis=1, keepdims=True)   # [steps, 1]
    noise_env_up = upsample_frames_numpy(noise_env, time_length)[:, 0]  # [time]
    noise_env_up = np.clip(noise_env_up, 0.0, 1.0)

    white_noise   = np.random.randn(time_length).astype(np.float32)
    audio_noise   = white_noise * noise_env_up

    # ── ミックス & tanh ───────────────────────────────────────────────
    audio = audio_harmonic + audio_noise * noise_gain
    audio = np.tanh(audio).astype(np.float32)
    return audio