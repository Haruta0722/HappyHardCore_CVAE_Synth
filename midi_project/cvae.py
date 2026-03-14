"""
cvae.py  ―  Conditional VAE モデル

【役割】
  ユーザー操作 (pitch, 音色ブレンド) を受け取り、
  DDSPパラメータを推論して dsp.py に渡す。

【ハード化時の使い方】
  VAE専用マイコンはこのファイルと config.py を参照する。
  DSP専用マイコンは dsp.py と config.py だけ参照すればよく、
  TensorFlow は不要。

【モジュール構成】
  TimbreEmbedding          音色ID → 連続埋め込みベクトル
  build_encoder()          Encoder q(z | x, cond)
  sample_z()               Reparameterization trick
  DDSPParameterDecoder     Decoder (z, cond) → DDSPパラメータ
  build_decoder()
  TimeWiseCVAE             VAE本体 (学習・推論・生成)
  ddsp_params_to_dict()    推論結果をマイコン送信用辞書に変換するヘルパー
"""

import numpy as np
import tensorflow as tf
from loss import Loss

from config import (
    SR,
    TIME_LENGTH,
    LATENT_DIM,
    LATENT_STEPS,
    NUM_HARMONICS,
    NOISE_FIR_LEN,
    TIMBRE_VOCAB,
    TIMBRE_EMBED_DIM,
    TIMBRE_NAMES,
    COND_DIM,
    ENCODER_CHANNELS,
)
from dsp import (
    HarmonicSynthesizer,
    FilteredNoiseSynthesizer,
    upsample_frames,
)


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
# TimbreEmbedding  音色ID → 連続埋め込みベクトル
# ============================================================
class TimbreEmbedding(tf.keras.layers.Layer):
    """
    音色カテゴリ ID (整数) を連続埋め込みベクトルに変換し、
    pitch と結合して条件ベクトル cond を作る。

    学習が進むにつれて音響的に近い音色は潜在空間でも近い位置に配置される。

    many-hot (音色ブレンド) への拡張:
        blend(pitch, timbre_weights) を使い、
        各カテゴリの埋め込みを重み付き平均することで任意の音色混合を表現する。
    """

    def __init__(self, vocab_size=TIMBRE_VOCAB, embed_dim=TIMBRE_EMBED_DIM):
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
            pitch     : 正規化ピッチ [batch]  float32
            timbre_id : 音色ID       [batch]  int32
        Returns:
            cond: [batch, COND_DIM]
        """
        emb = self.embedding(timbre_id)  # [batch, embed_dim]
        return tf.concat([pitch[:, None], emb], axis=-1)  # [batch, COND_DIM]

    def blend(self, pitch, timbre_weights):
        """
        many-hot 用: 重み付き混合埋め込みを生成する。

        Args:
            pitch          : [batch]               float32
            timbre_weights : [batch, TIMBRE_VOCAB] float32 (合計1.0推奨)
        Returns:
            cond: [batch, COND_DIM]
        """
        all_emb = self.embedding(
            tf.range(self.embedding.input_dim)
        )  # [vocab, embed_dim]
        mixed = tf.matmul(timbre_weights, all_emb)  # [batch, embed_dim]
        return tf.concat([pitch[:, None], mixed], axis=-1)  # [batch, COND_DIM]


# ============================================================
# Encoder  q(z | x, cond)
# ============================================================
def build_encoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM):
    """
    音声波形と条件ベクトルを受け取り z_mean, z_logvar を出力。

    cond の注入方法:
      Conv1D 特徴量 [batch, LATENT_STEPS, 512] に cond を
      RepeatVector でフレーム方向にブロードキャストして concat し、
      追加の Conv1D で融合する。
    """
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1), name="enc_audio")
    cond = tf.keras.Input(shape=(cond_dim,), name="enc_cond")

    x = x_in
    for ch, k, s in ENCODER_CHANNELS:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    # x: [batch, LATENT_STEPS, 512]

    cond_bc = tf.keras.layers.RepeatVector(LATENT_STEPS)(
        cond
    )  # [batch, LATENT_STEPS, cond_dim]
    x_cond = tf.keras.layers.Concatenate(axis=-1)(
        [x, cond_bc]
    )  # [batch, LATENT_STEPS, 512+cond_dim]
    x_cond = tf.keras.layers.Conv1D(512, 3, padding="same", activation="relu")(
        x_cond
    )

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
# DDSPParameterDecoder  (z, cond) → DDSPパラメータ
# ============================================================
class DDSPParameterDecoder(tf.keras.layers.Layer):
    """
    論文準拠のDecoder。音声波形は生成せず DDSPパラメータのみを出力する。

    出力パラメータ:
      amplitudes            [batch, LATENT_STEPS, 1]
      harmonic_distribution [batch, LATENT_STEPS, NUM_HARMONICS]  (softmax済み)
      noise_magnitudes      [batch, LATENT_STEPS, NOISE_FIR_LEN]  (sigmoid済み)

    ネットワーク構成: input_mlp → GRU → output_mlp → 各ヘッド (論文準拠)
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
    amps, harm, noise = DDSPParameterDecoder()(z_in, cond)
    return tf.keras.Model([z_in, cond], [amps, harm, noise], name="decoder")


# ============================================================
# TimeWiseCVAE  VAE本体
# ============================================================
class TimeWiseCVAE(tf.keras.Model):
    """
    DDSP-style Conditional VAE (Timbre Embedding版)

    外部インターフェース:
      train_step(data)
        data = (audio, pitch, timbre_id)

      generate(pitch, timbre_id, temperature)
        → audio [batch, TIME_LENGTH, 1]

      generate_blend(pitch, timbre_weights, temperature)
        → audio [batch, TIME_LENGTH, 1]

      infer_ddsp_params(pitch, timbre_id または timbre_weights)
        → DDSPパラメータ辞書  ← ハード化時にマイコンへ送るペイロード

      reconstruct(audio, pitch, timbre_id)
        → audio [batch, TIME_LENGTH, 1]
    """

    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.timbre_embedding = TimbreEmbedding()
        self.encoder = build_encoder(latent_dim, cond_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)
        # dsp.py の TFレイヤー版を使用 (学習・PC推論用)
        self.harmonic_synth = HarmonicSynthesizer()
        self.noise_synth = FilteredNoiseSynthesizer()

        self.steps_per_epoch = steps_per_epoch
        self.kl_warmup_epochs = 30
        self.kl_rampup_epochs = 100
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 1.0
        self.free_bits = 0.5

        self.z_std_ema = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.best_recon = tf.Variable(
            float("inf"), trainable=False, dtype=tf.float32
        )

    # ----------------------------------------------------------
    # 内部ヘルパー
    # ----------------------------------------------------------
    def _make_cond(self, pitch, timbre_id):
        return self.timbre_embedding(pitch, timbre_id)

    def _pitch_to_f0(self, pitch):
        """正規化ピッチ → Hz  (pitch ∈ [0,1], MIDI 36〜71)"""
        midi = pitch * 35.0 + 36.0
        return 440.0 * tf.pow(2.0, (midi - 69.0) / 12.0)

    def _synthesize(
        self, amplitudes, harmonic_distribution, noise_magnitudes, f0_hz
    ):
        """DDSPパラメータ → 音声波形 (TFレイヤー版)"""
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
        # Keras の model.fit() はデータセットの構造に応じて
        # inputs にタプル全体 or 先頭要素だけを渡す場合がある。
        # train_step で直接処理するため、call() はパススルーのみ。
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            audio, pitch, timbre_id = inputs
        else:
            # fit() から audio だけ渡された場合のフォールバック
            # (実際の推論は train_step / generate / reconstruct で行う)
            audio = inputs
            pitch = tf.zeros([tf.shape(audio)[0]], dtype=tf.float32)
            timbre_id = tf.zeros([tf.shape(audio)[0]], dtype=tf.int32)

        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder([audio, cond], training=training)
        z = sample_z(z_mean, z_logvar)
        amps, harm, noise = self.decoder([z, cond], training=training)
        x_hat = self._synthesize(amps, harm, noise, self._pitch_to_f0(pitch))
        return x_hat[:, :, None], z_mean, z_logvar

    # ----------------------------------------------------------
    # generate / generate_blend
    # ----------------------------------------------------------
    def generate(self, pitch, timbre_id, temperature=1.0):
        """N(0,I) サンプリングで音声を生成 (単一音色)"""
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
        """N(0,I) サンプリングで音声を生成 (音色ブレンド)"""
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
    # infer_ddsp_params  ← ハード化時のメインAPI
    # ----------------------------------------------------------
    def infer_ddsp_params(
        self,
        pitch,
        timbre_id=None,
        timbre_weights=None,
        temperature=1.0,
    ) -> dict:
        """
        DDSPパラメータを推論してマイコン送信用の辞書として返す。

        VAE専用マイコンが呼び出し、結果をDSP専用マイコンへ送信する。
        DSP専用マイコンは受け取った辞書を dsp.synthesize_numpy() に渡す。

        Args:
            pitch          : [batch] float32  正規化ピッチ
            timbre_id      : [batch] int32    音色ID (単一音色モード)
            timbre_weights : [batch, TIMBRE_VOCAB] float32  (ブレンドモード)
                             timbre_id と timbre_weights はどちらか一方を指定
            temperature    : float  サンプリング温度

        Returns:
            {
              "f0_hz":                  float,         スカラー
              "amplitudes":             np.ndarray,    [LATENT_STEPS, 1]
              "harmonic_distribution":  np.ndarray,    [LATENT_STEPS, NUM_HARMONICS]
              "noise_magnitudes":       np.ndarray,    [LATENT_STEPS, NOISE_FIR_LEN]
            }
        """
        if timbre_weights is not None:
            cond = self.timbre_embedding.blend(pitch, timbre_weights)
        elif timbre_id is not None:
            cond = self._make_cond(pitch, timbre_id)
        else:
            raise ValueError(
                "timbre_id か timbre_weights のどちらかを指定してください"
            )

        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        amps, harm, noise = self.decoder([z, cond], training=False)

        # f0_hz を計算 (バッチ先頭要素のスカラー)
        midi = float(pitch[0]) * 35.0 + 36.0
        f0_hz = float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))

        return {
            "f0_hz": f0_hz,
            "amplitudes": amps[0].numpy(),  # [LATENT_STEPS, 1]
            "harmonic_distribution": harm[
                0
            ].numpy(),  # [LATENT_STEPS, NUM_HARMONICS]
            "noise_magnitudes": noise[
                0
            ].numpy(),  # [LATENT_STEPS, NOISE_FIR_LEN]
        }

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
    # KLスケジュール
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
        # AudioDataset.build() が ((audio, pitch, timbre_id), dummy) の形で渡す
        (audio, pitch, timbre_id), _ = data

        with tf.GradientTape() as tape:
            cond = self._make_cond(pitch, timbre_id)

            z_mean, z_logvar = self.encoder([audio, cond], training=True)
            z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)
            z = tf.clip_by_value(sample_z(z_mean, z_logvar), -10.0, 10.0)

            amps, harm, noise = self.decoder([z, cond], training=True)

            f0_hz = self._pitch_to_f0(pitch)
            x_hat_audio = self._synthesize(amps, harm, noise, f0_hz)[
                :, :TIME_LENGTH
            ]

            x_target = tf.clip_by_value(tf.squeeze(audio, axis=-1), -1.0, 1.0)
            x_hat_sq = tf.clip_by_value(x_hat_audio, -1.0, 1.0)

            s = self._safe
            recon = s(tf.reduce_mean(tf.square(x_target - x_hat_sq)))
            stft_l, mel_l, _ = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )
            stft_l = s(stft_l)
            mel_l = s(mel_l)
            hf_l = s(compute_high_freq_emphasis_loss(x_target, x_hat_sq))
            flux_l = s(compute_spectral_flux_loss(x_target, x_hat_sq), 0.1)

            kl_per_dim = -0.5 * (
                1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            kl_mean = tf.reduce_mean(kl_per_dim, axis=[0, 1])
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


# ============================================================
# ヘルパー: DDSPパラメータ辞書をマイコン送信用にシリアライズ
# ============================================================
def ddsp_params_to_dict(params: dict) -> dict:
    """
    infer_ddsp_params() の出力をマイコン送信用に変換する。

    numpy配列 → Python リスト に変換することで
    JSON / MessagePack / UART 等での送信が可能になる。

    使用例:
        params  = model.infer_ddsp_params(pitch, timbre_id=timbre_id)
        payload = ddsp_params_to_dict(params)
        uart.send(msgpack.packb(payload))   # DSPマイコンへ送信
    """
    return {
        "f0_hz": float(params["f0_hz"]),
        "amplitudes": params["amplitudes"].tolist(),
        "harmonic_distribution": params["harmonic_distribution"].tolist(),
        "noise_magnitudes": params["noise_magnitudes"].tolist(),
    }
