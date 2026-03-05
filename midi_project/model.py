"""
DDSP-style Conditional VAE  (Timbre Embedding版)

条件ベクトルの構成:
  pitch     : スカラー (正規化済み MIDI ノート番号)
  timbre_id : 整数 (0=Screech, 1=Acid, 2=Pluck)
              → TimbreEmbedding で TIMBRE_EMBED_DIM 次元の連続ベクトルに変換
  cond      : [pitch(1), timbre_embed(TIMBRE_EMBED_DIM)] → COND_DIM 次元

アーキテクチャの流れ:
  (pitch, timbre_id)
      → TimbreEmbedding → cond [batch, COND_DIM]
                ↓
  Audio → Encoder(audio, cond) → z_mean, z_logvar
                                        ↓ reparameterization
          (z, cond) → DDSPParameterDecoder
                           ├─ amplitudes            [batch, LATENT_STEPS, 1]
                           ├─ harmonic_distribution [batch, LATENT_STEPS, NUM_HARMONICS]
                           └─ noise_magnitudes      [batch, LATENT_STEPS, NOISE_FIR_LEN]
                                        ↓ DSPモジュール (学習パラメータなし)
                           HarmonicSynthesizer + FilteredNoiseSynthesizer
                                        ↓
                                    x_hat (音声波形)
"""

import tensorflow as tf
from loss import Loss
import numpy as np

# ============================================================
# グローバル定数
# ============================================================
SR = 48000
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)  # 62400 サンプル
NUM_HARMONICS = 32
NOISE_FIR_LEN = 65  # FIRフィルタ長 (奇数推奨)
LATENT_DIM = 64

# 音色埋め込み
TIMBRE_VOCAB = 3  # Screech=0, Acid=1, Pluck=2
TIMBRE_EMBED_DIM = 8  # 埋め込み次元 (データ量1296に対して適切なサイズ)

# Encoder / Decoder が受け取る条件ベクトルの次元
# = pitch(1) + timbre_embed(TIMBRE_EMBED_DIM)
COND_DIM = 1 + TIMBRE_EMBED_DIM  # = 9

# Encoder ダウンサンプル倍率: 4×4×2×2 = 64
encoder_channels = [
    (64, 5, 4),
    (128, 5, 4),
    (256, 5, 2),
    (512, 3, 2),
]
LATENT_STEPS = TIME_LENGTH // 64  # 975


# ============================================================
# 損失関数ユーティリティ
# ============================================================
def compute_spectral_flux_loss(
    y_true, y_pred, frame_length=2048, hop_length=512
):
    """スペクトルフラックスの差を損失として計算"""

    def get_flux(audio):
        mag = tf.abs(
            tf.signal.stft(audio, frame_length, hop_length, frame_length)
        )
        diff = tf.maximum(mag[:, 1:, :] - mag[:, :-1, :], 0.0)
        return tf.reduce_mean(diff, axis=-1)

    return tf.reduce_mean(tf.abs(get_flux(y_true) - get_flux(y_pred)))


def compute_high_freq_emphasis_loss(y_true, y_pred, sr=SR, cutoff_freq=2000.0):
    """高周波帯域エネルギー差を重視した損失"""
    stft_true = tf.signal.stft(y_true, 2048, 512, 2048)
    stft_pred = tf.signal.stft(y_pred, 2048, 512, 2048)
    mag_true = tf.abs(stft_true)
    mag_pred = tf.abs(stft_pred)
    num_bins = tf.shape(mag_true)[-1]
    freq_bins = (
        tf.cast(tf.range(num_bins), tf.float32)
        * (sr / 2.0)
        / tf.cast(num_bins, tf.float32)
    )
    weight = tf.reshape(
        tf.sigmoid((freq_bins - cutoff_freq) / 500.0), [1, 1, -1]
    )
    return tf.reduce_mean(tf.square(mag_true - mag_pred) * (1.0 + weight * 5.0))


# ============================================================
# ユーティリティ: フレーム → サンプルのアップサンプル
# ============================================================
def upsample_frames(x, target_length):
    """[batch, frames, ch] → [batch, target_length, ch] (bilinear補間)"""
    x = tf.expand_dims(x, axis=2)
    x = tf.image.resize(x, [target_length, 1], method="bilinear")
    x = tf.squeeze(x, axis=2)
    return x


# ============================================================
# TimbreEmbedding
# 音色IDを連続埋め込みベクトルに変換し、pitchと結合して cond を作る
# ============================================================
class TimbreEmbedding(tf.keras.layers.Layer):
    """
    音色カテゴリ ID → 連続埋め込みベクトル

    学習が進むにつれて、音響的に似た音色は潜在空間で近い位置に、
    異なる音色は遠い位置に自動的に配置される。

    将来的なmany-hot (音色混合) への拡張:
        emb_s = self.embedding(0)  # Screech
        emb_a = self.embedding(1)  # Acid
        mixed = 0.5 * emb_s + 0.5 * emb_a  # 50%ブレンド
        cond  = tf.concat([pitch[:, None], mixed], axis=-1)
    """

    def __init__(
        self,
        vocab_size=TIMBRE_VOCAB,
        embed_dim=TIMBRE_EMBED_DIM,
    ):
        super().__init__(name="timbre_embedding")
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="timbre_embed_table",
        )

    def call(self, pitch, timbre_id):
        """
        Args:
            pitch     : 正規化ピッチ [batch]  (float32)
            timbre_id : 音色カテゴリID [batch] (int32)
        Returns:
            cond : [batch, COND_DIM]  (= 1 + embed_dim)
        """
        emb = self.embedding(timbre_id)  # [batch, embed_dim]
        cond = tf.concat([pitch[:, None], emb], axis=-1)  # [batch, 1+embed_dim]
        return cond

    def blend(self, pitch, timbre_weights):
        """
        many-hot 用: 重み付き混合埋め込みを生成する。

        Args:
            pitch          : [batch]               (float32)
            timbre_weights : [batch, TIMBRE_VOCAB] (float32, 合計1.0に正規化推奨)
        Returns:
            cond : [batch, COND_DIM]
        """
        # embedding テーブル全体 [vocab, embed_dim]
        all_emb = self.embedding(tf.range(self.embedding.input_dim))
        # 重み付き平均  [batch, embed_dim]
        mixed = tf.matmul(timbre_weights, all_emb)
        cond = tf.concat([pitch[:, None], mixed], axis=-1)
        return cond


# ============================================================
# Encoder  q(z | x, cond)
# ============================================================
def build_encoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM):
    """
    音声波形と条件ベクトルを受け取り z_mean, z_logvar を出力。

    cond の注入方法:
      Conv1D で抽出した特徴量 [batch, LATENT_STEPS, 512] に対して
      cond を RepeatVector でフレーム方向にブロードキャストして concat し、
      追加の Conv1D で融合する。
    """
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1), name="enc_audio")
    cond = tf.keras.Input(shape=(cond_dim,), name="enc_cond")

    # --- 音声特徴抽出 ---
    x = x_in
    for ch, k, s in encoder_channels:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    # x: [batch, LATENT_STEPS, 512]

    # --- cond をフレーム方向にブロードキャストして concat ---
    cond_bc = tf.keras.layers.RepeatVector(LATENT_STEPS)(
        cond
    )  # [batch, LATENT_STEPS, cond_dim]
    x_cond = tf.keras.layers.Concatenate(axis=-1)(
        [x, cond_bc]
    )  # [batch, LATENT_STEPS, 512+cond_dim]
    x_cond = tf.keras.layers.Conv1D(512, 3, padding="same", activation="relu")(
        x_cond
    )

    # --- 平均・分散 ---
    z_mean = tf.keras.layers.Conv1D(
        latent_dim, 3, padding="same", name="z_mean"
    )(x_cond)
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
        name="z_logvar_raw",
    )(x_cond)
    z_logvar = tf.keras.layers.Lambda(
        lambda v: tf.clip_by_value(v, -8.0, 2.0), name="z_logvar"
    )(z_logvar)

    return tf.keras.Model([x_in, cond], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    """Reparameterization trick"""
    return z_mean + tf.exp(0.5 * z_logvar) * tf.random.normal(tf.shape(z_mean))


# ============================================================
# Decoder  (z, cond) → DDSPパラメータ
# ============================================================
class DDSPParameterDecoder(tf.keras.layers.Layer):
    """
    論文準拠の Decoder。
    音声波形は生成せず、DSPモジュールへ渡すパラメータのみを出力する。

    出力:
      amplitudes            [batch, LATENT_STEPS, 1]
      harmonic_distribution [batch, LATENT_STEPS, NUM_HARMONICS]  (softmax済み)
      noise_magnitudes      [batch, LATENT_STEPS, NOISE_FIR_LEN]  (sigmoid済み)

    ネットワーク: input_mlp → GRU → output_mlp → 各ヘッド
    """

    def __init__(
        self, num_harmonics=NUM_HARMONICS, noise_fir_len=NOISE_FIR_LEN
    ):
        super().__init__(name="ddsp_param_decoder")
        self.num_harmonics = num_harmonics
        self.noise_fir_len = noise_fir_len

        self.input_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
            ],
            name="input_mlp",
        )

        self.gru = tf.keras.layers.GRU(512, return_sequences=True, name="gru")

        self.output_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
            ],
            name="output_mlp",
        )

        # 各パラメータへのヘッド
        self.amp_head = tf.keras.layers.Dense(1, name="amp_head")
        self.harm_head = tf.keras.layers.Dense(num_harmonics, name="harm_head")
        self.noise_head = tf.keras.layers.Dense(
            noise_fir_len, name="noise_head"
        )

    @staticmethod
    def _modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
        """論文の振幅活性化関数。ゼロ付近の解像度が高い。"""
        return max_value * tf.sigmoid(x) ** tf.math.log(exponent) + threshold

    def call(self, z, cond):
        latent_steps = tf.shape(z)[1]
        cond_bc = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        x = tf.concat([z, cond_bc], axis=-1)

        x = self.input_mlp(x)
        x = self.gru(x)
        x = self.output_mlp(x)

        amplitudes = self._modified_sigmoid(self.amp_head(x))
        harmonic_distribution = tf.nn.softmax(self.harm_head(x), axis=-1)
        noise_magnitudes = tf.sigmoid(self.noise_head(x))

        return amplitudes, harmonic_distribution, noise_magnitudes


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim), name="dec_z")
    cond = tf.keras.Input(shape=(cond_dim,), name="dec_cond")
    param_decoder = DDSPParameterDecoder()
    amps, harm_dist, noise_mag = param_decoder(z_in, cond)
    return tf.keras.Model(
        [z_in, cond], [amps, harm_dist, noise_mag], name="decoder"
    )


# ============================================================
# DSPモジュール  (学習パラメータなし・微分可能)
# ============================================================
class HarmonicSynthesizer(tf.keras.layers.Layer):
    """
    加算合成シンセサイザー (DDSP論文 Section 3.1)

    cumsum による累積位相計算でフレーム境界の位相連続性を保証する。
    """

    def __init__(self, sr=SR, time_length=TIME_LENGTH):
        super().__init__(name="harmonic_synth")
        self.sr = float(sr)
        self.time_length = time_length

    def call(self, f0_hz, amplitudes, harmonic_distribution):
        """
        Args:
            f0_hz                : [batch]
            amplitudes           : [batch, frames, 1]
            harmonic_distribution: [batch, frames, num_harmonics]
        Returns:
            audio: [batch, time_length]
        """
        num_harmonics = tf.shape(harmonic_distribution)[2]
        time_frames = tf.shape(amplitudes)[1]

        # 各倍音の絶対振幅
        harm_amps = (
            amplitudes * harmonic_distribution
        )  # [batch, frames, num_harmonics]

        # f0 → 倍音周波数  [batch, frames, num_harmonics]
        f0_frames = tf.reshape(f0_hz, [-1, 1, 1])
        f0_frames = tf.tile(f0_frames, [1, time_frames, 1])
        harm_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
        harm_nums = tf.reshape(harm_nums, [1, 1, -1])
        harm_freqs = f0_frames * harm_nums

        # フレーム → サンプルにアップサンプル
        harm_amps = upsample_frames(harm_amps, self.time_length)
        harm_freqs = upsample_frames(harm_freqs, self.time_length)

        harm_amps = tf.clip_by_value(harm_amps, 0.0, 2.0)
        harm_freqs = tf.clip_by_value(harm_freqs, 0.0, self.sr / 2.0)

        # 累積位相 → sin 波
        delta_phase = 2.0 * np.pi * harm_freqs / self.sr
        phase = tf.cumsum(delta_phase, axis=1)
        harmonics = harm_amps * tf.sin(phase)
        audio = tf.reduce_sum(harmonics, axis=-1)
        return tf.clip_by_value(audio, -10.0, 10.0)


class FilteredNoiseSynthesizer(tf.keras.layers.Layer):
    """
    ノイズシンセサイザー (DDSP論文 Section 3.2)

    ホワイトノイズを noise_magnitudes でフレームごとに振幅整形する。
    """

    def __init__(self, sr=SR, time_length=TIME_LENGTH):
        super().__init__(name="noise_synth")
        self.sr = sr
        self.time_length = time_length

    def call(self, noise_magnitudes):
        """
        Args:
            noise_magnitudes: [batch, frames, noise_fir_len]
        Returns:
            audio: [batch, time_length]
        """
        batch_size = tf.shape(noise_magnitudes)[0]
        noise = tf.random.normal([batch_size, self.time_length])

        # フレームごとの平均エネルギーをエンベロープとして使用
        envelope = tf.reduce_mean(
            noise_magnitudes, axis=-1, keepdims=True
        )  # [batch, frames, 1]
        envelope = upsample_frames(envelope, self.time_length)
        envelope = tf.squeeze(envelope, axis=-1)  # [batch, time_length]
        envelope = tf.clip_by_value(envelope, 0.0, 1.0)

        return tf.clip_by_value(noise * envelope, -2.0, 2.0)


# ============================================================
# TimeWiseCVAE  (モデル本体)
# ============================================================
class TimeWiseCVAE(tf.keras.Model):
    """
    DDSP-style Conditional VAE with Timbre Embedding

    外部インターフェース:
      train_step(data)
        data = (audio, pitch, timbre_id)
          audio     : [batch, TIME_LENGTH, 1]  float32
          pitch     : [batch]                  float32  (正規化済み)
          timbre_id : [batch]                  int32    (0/1/2)

      generate(pitch, timbre_id, temperature=1.0)
        → audio [batch, TIME_LENGTH, 1]

      reconstruct(audio, pitch, timbre_id)
        → audio [batch, TIME_LENGTH, 1]
    """

    def __init__(
        self,
        cond_dim=COND_DIM,
        latent_dim=LATENT_DIM,
        steps_per_epoch=87,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # --- サブモジュール ---
        self.timbre_embedding = TimbreEmbedding()
        self.encoder = build_encoder(latent_dim, cond_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
        self.harmonic_synth = HarmonicSynthesizer()
        self.noise_synth = FilteredNoiseSynthesizer()

        # --- KL ウォームアップスケジュール ---
        self.steps_per_epoch = steps_per_epoch
        self.kl_warmup_epochs = 30
        self.kl_rampup_epochs = 100
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 1.0
        self.free_bits = 0.5

        # --- 監視用変数 ---
        self.z_std_ema = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.best_recon = tf.Variable(
            float("inf"), trainable=False, dtype=tf.float32
        )

    # ----------------------------------------------------------
    # 内部ヘルパー
    # ----------------------------------------------------------
    def _make_cond(self, pitch, timbre_id):
        """pitch + timbre_id → cond ベクトル [batch, COND_DIM]"""
        return self.timbre_embedding(pitch, timbre_id)

    def _pitch_to_f0(self, pitch):
        """正規化ピッチ → Hz  (pitch ∈ [0,1], MIDI 36〜71)"""
        midi = pitch * 35.0 + 36.0
        return 440.0 * tf.pow(2.0, (midi - 69.0) / 12.0)

    def _synthesize(
        self, amplitudes, harmonic_distribution, noise_magnitudes, f0_hz
    ):
        """DDSPパラメータ → 音声波形"""
        audio = self.harmonic_synth(f0_hz, amplitudes, harmonic_distribution)
        audio += self.noise_synth(noise_magnitudes) * 0.1
        return tf.keras.activations.tanh(audio)

    @staticmethod
    def _safe(val, fallback=1.0):
        val = tf.where(tf.math.is_nan(val), tf.cast(fallback, val.dtype), val)
        val = tf.where(tf.math.is_inf(val), tf.cast(fallback, val.dtype), val)
        return val

    # ----------------------------------------------------------
    # call
    # ----------------------------------------------------------
    def call(self, inputs, training=None):
        """inputs = (audio, pitch, timbre_id)"""
        audio, pitch, timbre_id = inputs
        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder([audio, cond], training=training)
        z = sample_z(z_mean, z_logvar)
        amps, harm, noise = self.decoder([z, cond], training=training)
        x_hat = self._synthesize(amps, harm, noise, self._pitch_to_f0(pitch))
        return x_hat[:, :, None], z_mean, z_logvar

    # ----------------------------------------------------------
    # generate
    # ----------------------------------------------------------
    def generate(self, pitch, timbre_id, temperature=1.0):
        """
        事前分布 N(0, I) からサンプリングして音声を生成。

        Args:
            pitch       : [batch]  float32  正規化ピッチ
            timbre_id   : [batch]  int32    音色ID
            temperature : float   サンプリング温度
        Returns:
            audio : [batch, TIME_LENGTH, 1]
        """
        cond = self._make_cond(pitch, timbre_id)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        amps, harm, noise = self.decoder([z, cond], training=False)
        audio = self._synthesize(amps, harm, noise, self._pitch_to_f0(pitch))
        return audio[:, :, None]

    def generate_blend(self, pitch, timbre_weights, temperature=1.0):
        """
        many-hot 音色ブレンドで音声を生成。

        Args:
            pitch          : [batch]               float32
            timbre_weights : [batch, TIMBRE_VOCAB] float32  (合計1.0推奨)
            temperature    : float
        Returns:
            audio : [batch, TIME_LENGTH, 1]
        """
        cond = self.timbre_embedding.blend(pitch, timbre_weights)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        amps, harm, noise = self.decoder([z, cond], training=False)
        audio = self._synthesize(amps, harm, noise, self._pitch_to_f0(pitch))
        return audio[:, :, None]

    # ----------------------------------------------------------
    # encode / reconstruct
    # ----------------------------------------------------------
    def encode(self, audio, pitch, timbre_id):
        cond = self._make_cond(pitch, timbre_id)
        return self.encoder([audio, cond], training=False)

    def reconstruct(self, audio, pitch, timbre_id):
        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder([audio, cond], training=False)
        z = sample_z(z_mean, z_logvar)
        amps, harm, noise = self.decoder([z, cond], training=False)
        audio_out = self._synthesize(
            amps, harm, noise, self._pitch_to_f0(pitch)
        )
        return audio_out[:, :, None]

    # ----------------------------------------------------------
    # KL スケジュール
    # ----------------------------------------------------------
    def compute_kl_weight(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        done = tf.cast(step >= self.kl_warmup_steps, tf.float32)
        prog = tf.clip_by_value(
            (step - self.kl_warmup_steps) / self.kl_rampup_steps, 0.0, 1.0
        )
        return self.kl_target * prog * done

    # ----------------------------------------------------------
    # train_step
    # ----------------------------------------------------------
    def train_step(self, data):
        """
        data: (audio, pitch, timbre_id) のタプル
          audio     : [batch, TIME_LENGTH, 1]  float32
          pitch     : [batch]                  float32
          timbre_id : [batch]                  int32
        """
        audio, pitch, timbre_id = data

        with tf.GradientTape() as tape:
            # 条件ベクトル生成
            cond = self._make_cond(pitch, timbre_id)

            # エンコード
            z_mean, z_logvar = self.encoder([audio, cond], training=True)
            z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)
            z = tf.clip_by_value(sample_z(z_mean, z_logvar), -10.0, 10.0)

            # デコード → DDSPパラメータ
            amps, harm, noise = self.decoder([z, cond], training=True)

            # DSP合成
            f0_hz = self._pitch_to_f0(pitch)
            x_hat_audio = self._synthesize(amps, harm, noise, f0_hz)[
                :, :TIME_LENGTH
            ]

            x_target = tf.clip_by_value(tf.squeeze(audio, axis=-1), -1.0, 1.0)
            x_hat_sq = tf.clip_by_value(x_hat_audio, -1.0, 1.0)

            # 各損失
            s = self._safe  # エイリアス
            recon = s(tf.reduce_mean(tf.square(x_target - x_hat_sq)))
            stft_l, mel_l, _ = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )
            stft_l = s(stft_l)
            mel_l = s(mel_l)
            hf_l = s(compute_high_freq_emphasis_loss(x_target, x_hat_sq))
            flux_l = s(compute_spectral_flux_loss(x_target, x_hat_sq), 0.1)

            # KL (次元ごとに free_bits を適用して posterior collapse を防止)
            kl_per_dim = -0.5 * (
                1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            kl_mean = tf.reduce_mean(kl_per_dim, axis=[0, 1])  # [latent_dim]
            kl = s(
                tf.clip_by_value(
                    tf.reduce_mean(tf.maximum(kl_mean, self.free_bits)),
                    0.0,
                    100.0,
                ),
                0.5,
            )

            kl_w = self.compute_kl_weight()

            loss = (
                recon * 5.0
                + stft_l * 3.0
                + mel_l * 4.0
                + hf_l * 2.0
                + flux_l * 2.0
                + kl * kl_w
            )
            loss = s(loss, 1000.0)

        # 勾配更新
        grads = tape.gradient(loss, self.trainable_variables)
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
        grad_norm = tf.where(tf.math.is_nan(grad_norm), 0.0, grad_norm)
        grad_norm = tf.where(tf.math.is_inf(grad_norm), 0.0, grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 監視メトリクス更新
        z_std = s(tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1)), 1.0)
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)
        self.best_recon.assign(tf.minimum(self.best_recon, recon))

        return {
            "loss": loss,
            "recon": recon,
            "best_recon": self.best_recon,
            "stft": stft_l,
            "mel": mel_l,
            "high_freq": hf_l,
            "spectral_flux": flux_l,
            "kl": kl,
            "kl_weight": kl_w,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }
