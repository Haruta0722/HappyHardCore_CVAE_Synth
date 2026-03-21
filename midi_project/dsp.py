"""
dsp.py  ―  DDSPシンセサイザーモジュール (マルチエンベロープルーティング版)

【エンベロープルーティング設計】
  NUM_ENVELOPES = 4 本のADSRエンベロープを定義し、
  NUM_TARGETS   = 6 個のモジュールパラメータへのルーティングを
  ゲート行列 G [4×6] で制御する。

  適用先 (TARGET_NAMES):
    0: cutoff            フィルターカットオフ
    1: resonance         フィルターレゾナンス
    2: detune_cents      Unisonデチューン幅
    3: unison_blend      Unisonブレンド
    4: harmonic_bright   倍音の明暗 (高次倍音のスケール)
    5: noise_amount      ノイズ量

  合成時:
    env_values(t) = [env_0(t), ..., env_3(t)]        # [4]
    modulation(t) = G.T @ env_values(t)               # [6]
    param(t)      = base_param + modulation(t)[i]     # clamp(0,1)

【学習時の正則化】
  1. スパース正則化: g*(1-g) → ゲートを0か1に近づける
  2. エントロピー正則化: -var(env_i(t)) → エンベロープがフラットにならないよう
  3. Diversity Loss: cos_sim(env_i, env_j) → エンベロープ間を互いに異なる形にする
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

from config import SR, TIME_LENGTH, NUM_HARMONICS

# ============================================================
# エンベロープルーティング定数
# ============================================================
NUM_ENVELOPES = 4  # ADSRエンベロープの本数
NUM_TARGETS = 6  # 適用先モジュール数

TARGET_NAMES = [
    "cutoff",  # 0
    "resonance",  # 1
    "detune_cents",  # 2
    "unison_blend",  # 3
    "harmonic_bright",  # 4
    "noise_amount",  # 5
]


# ============================================================
# DDSPParams
# ============================================================
@dataclass
class DDSPParams:
    """
    DSPモジュール全体のパラメータ。

    エンベロープ関連:
      env_adsr    : [NUM_ENVELOPES, 4]  各エンベロープのA/D/S/R (0〜1)
      env_gates   : [NUM_ENVELOPES, NUM_TARGETS]  ゲート行列 (0〜1)
                    G[i][j] = エンベロープiをターゲットjに何割かける

    ベース値 (エンベロープがない状態の固定値):
      cutoff_base, resonance_base, detune_base,
      blend_base, brightness_base, noise_base
    """

    # Oscillator
    f0_hz: float = 440.0
    harmonic_amps: list[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (NUM_HARMONICS - 1)
    )

    # Unison (ベース値)
    unison_voices: int = 1
    detune_base: float = 0.0  # ベースデチューン 0〜1 (→ 0〜100cent)
    blend_base: float = 0.5  # ベースUnisonブレンド

    # ADSR (音量エンベロープ用・固定)
    attack: float = 0.1
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.3

    # Filter (ベース値)
    cutoff_base: float = 1.0
    resonance_base: float = 0.0

    # Noise (ベース値)
    noise_base: float = 0.0

    # 倍音明暗 (ベース値)
    brightness_base: float = 0.5  # 0=暗い(高次倍音抑制) 1=明るい(高次倍音強調)

    # マルチエンベロープ
    # env_adsr[i] = [attack, decay, sustain, release] for envelope i
    env_adsr: list = field(
        default_factory=lambda: [[0.1, 0.3, 0.5, 0.3]] * NUM_ENVELOPES
    )
    # env_gates[i][j] = gate value (0〜1)
    env_gates: list = field(
        default_factory=lambda: [
            [0.0] * NUM_TARGETS for _ in range(NUM_ENVELOPES)
        ]
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(d["harmonic_amps"], np.ndarray):
            d["harmonic_amps"] = d["harmonic_amps"].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DDSPParams":
        return cls(**d)

    def clamp(self) -> "DDSPParams":
        self.unison_voices = int(np.clip(self.unison_voices, 1, 7))
        self.detune_base = float(np.clip(self.detune_base, 0.0, 1.0))
        self.blend_base = float(np.clip(self.blend_base, 0.0, 1.0))
        self.attack = float(np.clip(self.attack, 0.0, 1.0))
        self.decay = float(np.clip(self.decay, 0.0, 1.0))
        self.sustain = float(np.clip(self.sustain, 0.0, 1.0))
        self.release = float(np.clip(self.release, 0.0, 1.0))
        self.cutoff_base = float(np.clip(self.cutoff_base, 0.0, 1.0))
        self.resonance_base = float(np.clip(self.resonance_base, 0.0, 1.0))
        self.noise_base = float(np.clip(self.noise_base, 0.0, 1.0))
        self.brightness_base = float(np.clip(self.brightness_base, 0.0, 1.0))
        self.harmonic_amps = list(np.clip(self.harmonic_amps, 0.0, 1.0))
        self.env_adsr = [
            [float(np.clip(v, 0.0, 1.0)) for v in row] for row in self.env_adsr
        ]
        self.env_gates = [
            [float(np.clip(v, 0.0, 1.0)) for v in row] for row in self.env_gates
        ]
        return self


# ============================================================
# ユーティリティ
# ============================================================
def adsr_to_seconds(value: float, total: float = TIME_LENGTH / SR) -> float:
    """正規化値 [0,1] → 秒数 (対数スケール)"""
    min_sec = 0.005
    max_sec = total * 0.5
    return float(min_sec * (max_sec / min_sec) ** float(value))


def cutoff_to_hz(cutoff: float, sr: int = SR) -> float:
    min_hz = 20.0
    max_hz = sr / 2.0
    return float(min_hz * (max_hz / min_hz) ** cutoff)


# ============================================================
# モジュール1: Oscillator
# ============================================================
def oscillator_numpy(
    f0_hz: float,
    harmonic_amps: np.ndarray,
    brightness: float = 0.5,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """
    加算合成。brightness で高次倍音のスケールを調整する。

    brightness=0.0: 高次倍音をさらに抑制 (暗い音)
    brightness=0.5: harmonic_amps をそのまま使用
    brightness=1.0: 高次倍音を強調 (明るい音)
    """
    harmonic_amps = np.array(harmonic_amps, dtype=np.float32)
    harmonic_amps = np.clip(harmonic_amps, 0.0, 1.0)
    harmonic_amps = np.where(harmonic_amps < 0.05, 0.0, harmonic_amps)
    num_harmonics = len(harmonic_amps)

    # brightness による倍音スケール調整
    # brightness=0.5 → scale=1.0 (変化なし)
    # brightness=0.0 → 高次倍音を追加減衰
    # brightness=1.0 → 高次倍音を追加ブースト
    harm_idx = np.arange(num_harmonics, dtype=np.float32)
    bright_mod = 1.0 + (brightness - 0.5) * 2.0 * (
        harm_idx / (num_harmonics - 1)
    )
    bright_mod = np.clip(bright_mod, 0.0, 2.0)
    amps = np.clip(harmonic_amps * bright_mod, 0.0, 1.0)

    harm_nums = np.arange(1, num_harmonics + 1, dtype=np.float32)
    harm_freqs = np.clip(f0_hz * harm_nums, 0.0, sr / 2.0)

    delta_phase = 2.0 * np.pi * harm_freqs[None, :] / sr
    phase = np.cumsum(np.tile(delta_phase, (time_length, 1)), axis=0)
    audio = (amps[None, :] * np.sin(phase)).sum(axis=1)
    return audio.astype(np.float32)


# ============================================================
# モジュール1b: Unison
# ============================================================
def unison_numpy(
    f0_hz: float,
    harmonic_amps: np.ndarray,
    brightness: float = 0.5,
    unison_voices: int = 1,
    detune_cents: float = 0.0,
    unison_blend: float = 0.5,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    n = max(1, int(unison_voices))
    if n == 1 or detune_cents < 0.001:
        return oscillator_numpy(
            f0_hz, harmonic_amps, brightness, time_length, sr
        )

    dry = oscillator_numpy(f0_hz, harmonic_amps, brightness, time_length, sr)
    offsets = np.linspace(-detune_cents / 2.0, detune_cents / 2.0, n)
    wet = np.zeros(time_length, dtype=np.float32)
    for cents in offsets:
        wet += oscillator_numpy(
            f0_hz * (2.0 ** (cents / 1200.0)),
            harmonic_amps,
            brightness,
            time_length,
            sr,
        )
    wet /= n
    blend = float(np.clip(unison_blend, 0.0, 1.0))
    return ((1.0 - blend) * dry + blend * wet).astype(np.float32)


# ============================================================
# モジュール2: ADSR Envelope (スカラー出力)
# ============================================================
def adsr_envelope_numpy_fast(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """ADSRエンベロープ [time_length] を生成する"""
    t = np.linspace(
        0.0, time_length / sr, time_length, endpoint=False, dtype=np.float32
    )
    total_t = time_length / sr
    a_end = float(attack)
    d_end = float(attack + decay)
    r_start = float(max(d_end, total_t - release))

    env = np.where(
        t < a_end,
        t / (a_end + 1e-6),
        np.where(
            t < d_end,
            1.0 - (1.0 - sustain) * (t - a_end) / (decay + 1e-6),
            np.where(
                t < r_start,
                np.full_like(t, sustain),
                sustain
                * np.clip(1.0 - (t - r_start) / (release + 1e-6), 0.0, 1.0),
            ),
        ),
    )
    return np.clip(env, 0.0, 1.0).astype(np.float32)


# ============================================================
# モジュール2b: マルチエンベロープルーティング
# ============================================================
def compute_modulations(
    env_adsr: list,  # [NUM_ENVELOPES, 4]  各エンベロープのA/D/S/R
    env_gates: list,  # [NUM_ENVELOPES, NUM_TARGETS]  ゲート行列
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """
    ゲート行列を使って各ターゲットへのモジュレーション量を計算する。

    Returns:
        modulations: [time_length, NUM_TARGETS]
                     各時刻における各ターゲットへの変調量 (0〜1)
    """
    gates = np.array(env_gates, dtype=np.float32)  # [NUM_ENV, NUM_TARGETS]
    mods = np.zeros((time_length, NUM_TARGETS), dtype=np.float32)

    for i, adsr in enumerate(env_adsr):
        a_sec = adsr_to_seconds(adsr[0], time_length / sr)
        d_sec = adsr_to_seconds(adsr[1], time_length / sr)
        s_lv = float(adsr[2])
        r_sec = adsr_to_seconds(adsr[3], time_length / sr)

        env_curve = adsr_envelope_numpy_fast(
            a_sec, d_sec, s_lv, r_sec, time_length, sr
        )
        # env_curve: [time_length]
        # gates[i]: [NUM_TARGETS]
        mods += env_curve[:, None] * gates[i][None, :]  # [time, 6]

    return np.clip(mods, 0.0, 1.0)


# ============================================================
# モジュール3: SVF Filter (時変対応)
# ============================================================
def svf_filter_numpy(
    audio: np.ndarray,
    cutoff: float,
    resonance: float,
    mode: str = "lowpass",
    sr: int = SR,
) -> np.ndarray:
    cutoff_hz = cutoff_to_hz(cutoff, sr)
    f = float(np.clip(2.0 * np.sin(np.pi * cutoff_hz / sr), 0.0, 1.0))
    q = 1.0 - resonance * 0.99
    out = np.zeros_like(audio)
    lp = bp = 0.0
    for i in range(len(audio)):
        hp = float(audio[i]) - lp - q * bp
        bp = f * hp + bp
        lp = f * bp + lp
        out[i] = lp if mode == "lowpass" else (hp if mode == "highpass" else bp)
    return np.clip(out, -2.0, 2.0).astype(np.float32)


def svf_filter_numpy_fast(
    audio: np.ndarray,
    cutoff: float,
    resonance: float,
    mode: str = "lowpass",
    sr: int = SR,
) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt

        cutoff_hz = float(
            np.clip(cutoff_to_hz(cutoff, sr), 20.0, sr / 2.0 - 1.0)
        )
        nyq = sr / 2.0
        norm = cutoff_hz / nyq
        q = float(np.clip(0.5 + resonance * 9.5, 0.5, 10.0))
        if mode == "lowpass":
            sos = butter(2, norm, btype="low", output="sos")
        elif mode == "highpass":
            sos = butter(2, norm, btype="high", output="sos")
        else:
            bw = norm / q
            sos = butter(
                2,
                [max(1e-4, norm - bw / 2), min(0.9999, norm + bw / 2)],
                btype="band",
                output="sos",
            )
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


# ============================================================
# モジュール4: Noise
# ============================================================
def noise_generator_numpy(
    noise_amount: float, time_length: int = TIME_LENGTH, seed: int = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return (
        np.random.randn(time_length) * float(np.clip(noise_amount, 0.0, 1.0))
    ).astype(np.float32)


# ============================================================
# メイン合成関数
# ============================================================
def synthesize_numpy(
    params: DDSPParams,
    sr: int = SR,
    time_length: int = TIME_LENGTH,
    fast_filter: bool = True,
    seed: int = None,
) -> np.ndarray:
    """
    DDSPParams から音声波形を合成する。

    処理フロー:
      1. マルチエンベロープルーティングで時変パラメータを計算
      2. Oscillator + Unison (時変brightness/detune/blend)
      3. 音量ADSR エンベロープ
      4. Filter (時変cutoff/resonance)
      5. Noise (時変noise_amount)
      6. tanh クリッピング
    """
    params = params.clamp()

    # --- 1. モジュレーション計算 [time, NUM_TARGETS] ---
    mods = compute_modulations(
        params.env_adsr, params.env_gates, time_length, sr
    )
    # TARGET: 0=cutoff, 1=resonance, 2=detune, 3=blend, 4=brightness, 5=noise

    # --- 2. Oscillator + Unison (時変パラメータ適用) ---
    # brightnessは時間変化させるためフレームごとに合成は重いので
    # 時間平均値を使用 (GUIで十分な精度)
    brightness_t = float(
        np.clip(params.brightness_base + mods[:, 4].mean(), 0.0, 1.0)
    )
    detune_t = float(
        np.clip((params.detune_base + mods[:, 2].mean()) * 100.0, 0.0, 100.0)
    )
    blend_t = float(np.clip(params.blend_base + mods[:, 3].mean(), 0.0, 1.0))

    audio = unison_numpy(
        f0_hz=params.f0_hz,
        harmonic_amps=np.array(params.harmonic_amps, dtype=np.float32),
        brightness=brightness_t,
        unison_voices=params.unison_voices,
        detune_cents=detune_t,
        unison_blend=blend_t,
        time_length=time_length,
        sr=sr,
    )

    # --- 3. 音量ADSRエンベロープ ---
    a_sec = adsr_to_seconds(params.attack, time_length / sr)
    d_sec = adsr_to_seconds(params.decay, time_length / sr)
    r_sec = adsr_to_seconds(params.release, time_length / sr)
    envelope = adsr_envelope_numpy_fast(
        a_sec, d_sec, params.sustain, r_sec, time_length, sr
    )
    audio = audio * envelope

    # --- 4. Filter (時変cutoff/resonance) ---
    # cutoff と resonance は時間変化させるため、チャンクごとにフィルタリング
    cutoff_t = np.clip(params.cutoff_base + mods[:, 0], 0.0, 1.0)
    resonance_t = np.clip(params.resonance_base + mods[:, 1], 0.0, 1.0)

    # 時変フィルタ: チャンク処理 (変化が緩やかな場合の近似)
    CHUNK = 512
    filtered = np.zeros_like(audio)
    for start in range(0, time_length, CHUNK):
        end = min(start + CHUNK, time_length)
        c_avg = float(cutoff_t[start:end].mean())
        r_avg = float(resonance_t[start:end].mean())
        chunk = audio[start:end]
        if c_avg < 0.999:
            if fast_filter:
                chunk = svf_filter_numpy_fast(chunk, c_avg, r_avg, sr=sr)
            else:
                chunk = svf_filter_numpy(chunk, c_avg, r_avg, sr=sr)
        filtered[start:end] = chunk
    audio = filtered

    # --- 5. Noise (時変noise_amount) ---
    noise_t = np.clip(params.noise_base + mods[:, 5], 0.0, 1.0)
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(time_length).astype(np.float32)
    audio = audio + noise * noise_t

    # --- 6. tanh ---
    return np.tanh(audio).astype(np.float32)


# ============================================================
# 正則化損失 (学習時に cvae.py から呼ぶ)
# ============================================================
if HAS_TF:

    def envelope_regularization_losses(
        env_adsr_tf: tf.Tensor,  # [batch, NUM_ENV, 4]
        env_gates_tf: tf.Tensor,  # [batch, NUM_ENV, NUM_TARGETS]
        time_length: int = TIME_LENGTH,
        sr: int = SR,
        lambda_sparse: float = 0.01,
        lambda_entropy: float = 0.01,
        lambda_diverse: float = 0.005,
    ) -> dict:
        """
        エンベロープルーティングの正則化損失を計算する。

        1. スパース正則化: ゲートを 0 か 1 に近づける
           loss = mean(g * (1-g))   → g=0.5 で最大、g=0/1 で 0

        2. エントロピー正則化: エンベロープがフラットにならないよう
           loss = -mean(var(env_i(t)))  → 変化が大きいほど損失が小さい

        3. Diversity Loss: エンベロープ間のコサイン類似度を小さくする
           loss = mean(cos_sim(env_i, env_j)) for i≠j
        """
        total_t = time_length / sr
        min_sec = 0.005
        max_sec = total_t * 0.5

        def to_sec(v):
            return min_sec * tf.pow(max_sec / min_sec, v)

        t = tf.cast(tf.linspace(0.0, total_t, time_length), tf.float32)  # [T]

        # エンベロープカーブを計算 [batch, NUM_ENV, time]
        env_curves = []
        for i in range(NUM_ENVELOPES):
            a = tf.reshape(to_sec(env_adsr_tf[:, i, 0]), [-1, 1])
            d = tf.reshape(to_sec(env_adsr_tf[:, i, 1]), [-1, 1])
            s = tf.reshape(env_adsr_tf[:, i, 2], [-1, 1])
            r = tf.reshape(to_sec(env_adsr_tf[:, i, 3]), [-1, 1])

            a_end = a
            d_end = a + d
            r_start = tf.maximum(d_end, total_t - r)
            t_bc = tf.reshape(t, [1, -1])  # [1, T]

            curve = tf.where(
                t_bc < a_end,
                t_bc / (a_end + 1e-6),
                tf.where(
                    t_bc < d_end,
                    1.0 - (1.0 - s) * (t_bc - a_end) / (d + 1e-6),
                    tf.where(
                        t_bc < r_start,
                        s * tf.ones_like(t_bc),
                        s
                        * tf.clip_by_value(
                            1.0 - (t_bc - r_start) / (r + 1e-6), 0.0, 1.0
                        ),
                    ),
                ),
            )  # [batch, T]
            env_curves.append(tf.clip_by_value(curve, 0.0, 1.0))

        env_stack = tf.stack(env_curves, axis=1)  # [batch, NUM_ENV, T]

        # --- 1. スパース正則化 ---
        g = env_gates_tf  # [batch, NUM_ENV, NUM_TARGETS]
        sparse_loss = tf.reduce_mean(g * (1.0 - g))

        # --- 2. エントロピー正則化 (エンベロープの分散を最大化) ---
        env_var = tf.math.reduce_variance(
            env_stack, axis=-1
        )  # [batch, NUM_ENV]
        entropy_loss = -tf.reduce_mean(env_var)

        # --- 3. Diversity Loss (エンベロープ間のコサイン類似度を最小化) ---
        # env_stack: [batch, NUM_ENV, T]
        norm = (
            tf.norm(env_stack, axis=-1, keepdims=True) + 1e-8
        )  # [batch, NUM_ENV, 1]
        normed = env_stack / norm  # [batch, NUM_ENV, T]
        # コサイン類似度行列 [batch, NUM_ENV, NUM_ENV]
        cos_sim = tf.matmul(normed, tf.transpose(normed, [0, 2, 1]))
        # 対角成分 (自己類似度=1) を除いた平均
        mask = 1.0 - tf.eye(NUM_ENVELOPES)
        diverse_loss = tf.reduce_mean(cos_sim * mask)

        total = (
            lambda_sparse * sparse_loss
            + lambda_entropy * entropy_loss
            + lambda_diverse * diverse_loss
        )
        return {
            "env_total": total,
            "env_sparse": sparse_loss,
            "env_entropy": entropy_loss,
            "env_diverse": diverse_loss,
        }

    def upsample_frames(x, target_length):
        x = tf.expand_dims(x, axis=2)
        x = tf.image.resize(x, [target_length, 1], method="bilinear")
        x = tf.squeeze(x, axis=2)
        return x

    class OscillatorLayer(tf.keras.layers.Layer):
        def __init__(self, sr=SR, time_length=TIME_LENGTH):
            super().__init__(name="oscillator")
            self.sr = float(sr)
            self.time_length = time_length

        def call(self, f0_hz, harmonic_amps, brightness=None):
            """
            f0_hz:         [batch]
            harmonic_amps: [batch, NUM_HARMONICS]
            brightness:    [batch]  0〜1 (Noneなら0.5固定)
            """
            num_harmonics = tf.shape(harmonic_amps)[1]
            harm_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)

            if brightness is not None:
                # brightness による倍音スケール [batch, H]
                harm_idx = tf.cast(tf.range(num_harmonics), tf.float32)
                bright_mod = 1.0 + (
                    tf.reshape(brightness, [-1, 1]) - 0.5
                ) * 2.0 * (harm_idx / tf.cast(num_harmonics - 1, tf.float32))
                bright_mod = tf.clip_by_value(bright_mod, 0.0, 2.0)
                amps = tf.clip_by_value(harmonic_amps * bright_mod, 0.0, 1.0)
            else:
                amps = harmonic_amps

            harm_freqs = tf.reshape(f0_hz, [-1, 1]) * tf.reshape(
                harm_nums, [1, -1]
            )
            harm_freqs = tf.clip_by_value(harm_freqs, 0.0, self.sr / 2.0)

            delta = (
                2.0
                * np.pi
                * tf.reshape(harm_freqs, [-1, 1, num_harmonics])
                / self.sr
            )
            delta = tf.tile(delta, [1, self.time_length, 1])
            phase = tf.cumsum(delta, axis=1)
            audio = tf.reduce_sum(
                tf.reshape(amps, [-1, 1, num_harmonics]) * tf.sin(phase),
                axis=-1,
            )
            return tf.clip_by_value(audio, -10.0, 10.0)

    class ADSRLayer(tf.keras.layers.Layer):
        def __init__(self, sr=SR, time_length=TIME_LENGTH):
            super().__init__(name="adsr")
            self.sr = float(sr)
            self.time_length = time_length
            self.total_t = time_length / sr

        def call(self, attack, decay, sustain, release):
            min_sec = 0.005
            max_sec = self.total_t * 0.5

            def to_sec(v):
                return min_sec * tf.pow(max_sec / min_sec, v)

            a = tf.reshape(to_sec(attack), [-1, 1])
            d = tf.reshape(to_sec(decay), [-1, 1])
            s = tf.reshape(sustain, [-1, 1])
            r = tf.reshape(to_sec(release), [-1, 1])
            t = tf.reshape(
                tf.cast(
                    tf.linspace(0.0, self.total_t, self.time_length), tf.float32
                ),
                [1, -1],
            )

            a_end = a
            d_end = a + d
            r_start = tf.maximum(d_end, self.total_t - r)

            env = tf.where(
                t < a_end,
                t / (a_end + 1e-6),
                tf.where(
                    t < d_end,
                    1.0 - (1.0 - s) * (t - a_end) / (d + 1e-6),
                    tf.where(
                        t < r_start,
                        s * tf.ones_like(t),
                        s
                        * tf.clip_by_value(
                            1.0 - (t - r_start) / (r + 1e-6), 0.0, 1.0
                        ),
                    ),
                ),
            )
            return tf.clip_by_value(env, 0.0, 1.0)

    class FilterLayer(tf.keras.layers.Layer):
        def __init__(self, sr=SR, time_length=TIME_LENGTH, n_fft=2048):
            super().__init__(name="filter")
            self.sr = float(sr)
            self.time_length = time_length
            self.n_fft = n_fft

        def call(self, audio, cutoff, resonance):
            min_hz = 20.0
            max_hz = self.sr / 2.0
            cutoff_hz = min_hz * tf.pow(max_hz / min_hz, cutoff)

            stft = tf.signal.stft(
                audio, self.n_fft, self.n_fft // 4, self.n_fft
            )
            n_bins = tf.shape(stft)[-1]
            freqs = tf.cast(tf.range(n_bins), tf.float32) * (
                self.sr / self.n_fft
            )

            cutoff_bc = tf.reshape(cutoff_hz, [-1, 1, 1])
            res_bc = tf.reshape(resonance, [-1, 1, 1])
            mask = tf.sigmoid((cutoff_bc - freqs) / (cutoff_bc * 0.1 + 1.0))
            peak = res_bc * tf.exp(
                -tf.square((freqs - cutoff_bc) / (cutoff_bc * 0.05 + 1.0))
            )
            mask = tf.clip_by_value(mask + peak * (1.0 - mask), 0.0, 2.0)

            filtered = tf.signal.inverse_stft(
                stft * tf.cast(mask, tf.complex64),
                self.n_fft,
                self.n_fft // 4,
                self.n_fft,
            )
            filtered = filtered[:, : self.time_length]
            filtered = tf.pad(
                filtered,
                [[0, 0], [0, self.time_length - tf.shape(filtered)[1]]],
            )
            return tf.clip_by_value(filtered, -10.0, 10.0)
