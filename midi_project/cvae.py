"""
cvae.py  ―  Conditional VAE (モジュール分離DDSP版)

【Decoderの出力パラメータ】
  Oscillator : harmonic_amps [NUM_HARMONICS]  各倍音の相対振幅
  ADSR       : attack, decay, sustain, release  各 0〜1
  Filter     : cutoff, resonance  各 0〜1
  Noise      : noise_amount  0〜1

  ※ f0_hz は pitch から直接計算するため Decoder は出力しない

【ハード化時のフロー】
  VAEマイコン:
    infer_ddsp_params(pitch, timbre_weights) → DDSPParams
    DDSPParams.to_dict() → 辞書を UART 等で送信

  DSPマイコン:
    DDSPParams.from_dict(received) → synthesize_numpy(params)
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
    COND_DIM,
    ENCODER_CHANNELS,
    TIMBRE_VOCAB,
    TIMBRE_EMBED_DIM,
)
from dsp import (
    DDSPParams,
    OscillatorLayer,
    ADSRLayer,
    FilterLayer,
    upsample_frames,
)


# ============================================================
# 音色ごとの倍音プロファイルテンプレート
# ============================================================
def _make_harmonic_templates(num_harmonics: int) -> tf.Tensor:
    """
    各音色カテゴリの「理想的な倍音プロファイル」を定義する。

    Screech : 高次倍音が強い (鋸歯状波に近い)
    Acid    : 中域倍音にピーク (TB-303的なフィルタースイープ音)
    Pluck   : 基音が強く指数的に減衰 (弦楽器の撥音)

    これをDecoderが出力する harmonic_amps の正解値として使い、
    音色カテゴリごとに異なる倍音分布を学習させる。

    Returns:
        templates: [TIMBRE_VOCAB, num_harmonics]  float32
    """
    H = num_harmonics
    idx = np.arange(1, H + 1, dtype=np.float32)

    # Screech (id=0): 奇数倍音強調 + 高域を保持 (鋸歯状 + クリップ)
    screech = 1.0 / idx  # 1/n ロールオフ
    screech[1::2] *= 0.3  # 偶数倍音を抑制
    screech = screech / screech.max()

    # Acid (id=1): 基音〜中域にエネルギー集中 (低次〜中次にピーク)
    peak = H * 0.25  # 全倍音数の1/4あたりにピーク
    acid = np.exp(-0.5 * ((idx - peak) / (H * 0.15)) ** 2)
    acid = acid / acid.max()

    # Pluck (id=2): 基音が最強、指数減衰
    pluck = np.exp(-idx * 0.18)
    pluck = pluck / pluck.max()

    templates = np.stack([screech, acid, pluck], axis=0).astype(np.float32)
    return tf.constant(templates, dtype=tf.float32)


# グローバルにテンプレートを作成
HARMONIC_TEMPLATES = _make_harmonic_templates(NUM_HARMONICS)


def compute_harmonic_timbre_loss(
    harmonic_amps: tf.Tensor,  # [batch, NUM_HARMONICS]
    timbre_id: tf.Tensor,  # [batch]  int32
) -> tf.Tensor:
    """
    Decoderが出力した harmonic_amps を、
    音色カテゴリに対応するテンプレートに近づける補助損失。

    timbre_id から対応するテンプレートを引いて MSE を計算する。
    これにより「Screech のときは高次倍音が強い」という
    音色固有の倍音構造を学習させる。
    """
    # timbre_id → テンプレート  [batch, NUM_HARMONICS]
    targets = tf.gather(HARMONIC_TEMPLATES, timbre_id)
    loss = tf.reduce_mean(tf.square(harmonic_amps - targets))
    return loss


# ============================================================
# 損失関数ユーティリティ
# ============================================================
def compute_spectral_flux_loss(
    y_true, y_pred, frame_length=2048, hop_length=512
):
    def get_flux(audio):
        mag = tf.abs(
            tf.signal.stft(audio, frame_length, hop_length, frame_length)
        )
        diff = tf.maximum(mag[:, 1:, :] - mag[:, :-1, :], 0.0)
        return tf.reduce_mean(diff, axis=-1)

    return tf.reduce_mean(tf.abs(get_flux(y_true) - get_flux(y_pred)))


def compute_high_freq_emphasis_loss(y_true, y_pred, sr=SR, cutoff_freq=2000.0):
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
# TimbreEmbedding
# ============================================================
class TimbreEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=TIMBRE_VOCAB, embed_dim=TIMBRE_EMBED_DIM):
        super().__init__(name="timbre_embedding")
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="timbre_embed_table",
        )

    def call(self, pitch, timbre_id):
        emb = self.embedding(timbre_id)
        return tf.concat([pitch[:, None], emb], axis=-1)

    def blend(self, pitch, timbre_weights):
        all_emb = self.embedding(tf.range(self.embedding.input_dim))
        mixed = tf.matmul(timbre_weights, all_emb)
        return tf.concat([pitch[:, None], mixed], axis=-1)


# ============================================================
# Encoder  q(z | x, cond)
# ============================================================
def build_encoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1), name="enc_audio")
    cond = tf.keras.Input(shape=(cond_dim,), name="enc_cond")

    x = x_in
    for ch, k, s in ENCODER_CHANNELS:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    cond_bc = tf.keras.layers.RepeatVector(LATENT_STEPS)(cond)
    x_cond = tf.keras.layers.Concatenate(axis=-1)([x, cond_bc])
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
    return z_mean + tf.exp(0.5 * z_logvar) * tf.random.normal(tf.shape(z_mean))


# ============================================================
# DDSPParameterDecoder  (z, cond) → DDSPパラメータ (分離版)
# ============================================================
class DDSPParameterDecoder(tf.keras.layers.Layer):
    """
    (z, cond) → DDSPパラメータを出力するDecoder。

    【出力パラメータ (すべてスカラー or 1Dベクトル, バッチ次元あり)】
      harmonic_amps  : [batch, NUM_HARMONICS]  各倍音の相対振幅 (softmax)
      attack         : [batch]  ADSRアタック   0〜1
      decay          : [batch]  ADSRディケイ   0〜1
      sustain        : [batch]  ADSRサステイン 0〜1
      release        : [batch]  ADSRリリース   0〜1
      cutoff         : [batch]  フィルタカットオフ 0〜1
      resonance      : [batch]  フィルタレゾナンス 0〜1
      noise_amount   : [batch]  ノイズ量       0〜1

    フレームごとの時系列ではなくサンプル全体を代表する1つの値を出力する。
    これによりGUIでの操作・可視化と1対1で対応する。
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__(name="ddsp_param_decoder")
        self.num_harmonics = num_harmonics

        # z の時系列を集約するGRU
        self.gru = tf.keras.layers.GRU(512, return_sequences=False, name="gru")

        # 共通特徴MLP
        self.shared_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
            ],
            name="shared_mlp",
        )

        # ── 各パラメータへの独立ヘッド ──────────────────────────────
        # Oscillator
        self.harm_head = tf.keras.layers.Dense(num_harmonics, name="harm_head")

        # ADSR (各パラメータを独立したヘッドで出力)
        self.attack_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="attack_head"
        )
        self.decay_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="decay_head"
        )
        self.sustain_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="sustain_head"
        )
        self.release_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="release_head"
        )

        # Filter
        self.cutoff_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="cutoff_head"
        )
        self.resonance_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="resonance_head"
        )

        # Noise
        self.noise_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="noise_head"
        )

    def call(self, z, cond):
        """
        Args:
            z:    [batch, LATENT_STEPS, latent_dim]
            cond: [batch, COND_DIM]
        Returns:
            辞書形式でDDSPパラメータを返す
        """
        # z と cond を結合して GRU に入力
        latent_steps = tf.shape(z)[1]
        cond_bc = tf.tile(cond[:, None, :], [1, latent_steps, 1])
        x = tf.concat([z, cond_bc], axis=-1)  # [batch, steps, latent+cond]

        # GRU で時系列を集約 → [batch, 512]
        x = self.gru(x)

        # 共通特徴
        feat = self.shared_mlp(x)  # [batch, 256]

        # 各パラメータを独立ヘッドで推論
        # softmax → sigmoid に変更:
        #   softmax は合計1の制約があり「全体の倍音量」を変えられない。
        #   sigmoid にすると各倍音が独立に 0〜1 を取れ、
        #   音色ごとに「倍音の豊かさ」の違いを表現できる。
        harmonic_amps = tf.sigmoid(self.harm_head(feat))  # [batch, H]
        attack = tf.squeeze(self.attack_head(feat), axis=-1)  # [batch]
        decay = tf.squeeze(self.decay_head(feat), axis=-1)
        sustain = tf.squeeze(self.sustain_head(feat), axis=-1)
        release = tf.squeeze(self.release_head(feat), axis=-1)
        cutoff = tf.squeeze(self.cutoff_head(feat), axis=-1)
        resonance = tf.squeeze(self.resonance_head(feat), axis=-1)
        noise_amount = tf.squeeze(self.noise_head(feat), axis=-1)

        return {
            "harmonic_amps": harmonic_amps,
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
            "cutoff": cutoff,
            "resonance": resonance,
            "noise_amount": noise_amount,
        }


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim), name="dec_z")
    cond = tf.keras.Input(shape=(cond_dim,), name="dec_cond")
    params = DDSPParameterDecoder()(z_in, cond)
    # Keras Model は辞書出力に対応している
    return tf.keras.Model([z_in, cond], params, name="decoder")


# ============================================================
# TimeWiseCVAE  VAE本体
# ============================================================
class TimeWiseCVAE(tf.keras.Model):
    """
    DDSP-style Conditional VAE (モジュール分離版)

    Decoderは Oscillator / ADSR / Filter / Noise のパラメータを独立して出力。
    GUIから各パラメータを直接操作・可視化できる。
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

        # DSPレイヤー (TF版・学習用)
        self.oscillator = OscillatorLayer()
        self.adsr = ADSRLayer()
        self.filter_l = FilterLayer()

        # KLスケジュール
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
        """正規化ピッチ [0,1] → Hz (MIDI 36〜71)"""
        midi = pitch * 35.0 + 36.0
        return 440.0 * tf.pow(2.0, (midi - 69.0) / 12.0)

    def _synthesize_from_params(self, p: dict, f0_hz, training=False):
        """
        Decoderが出力したパラメータ辞書 → 音声波形 (TF版)

        p: DDSPParameterDecoder の出力辞書
        """
        # 1. Oscillator
        audio = self.oscillator(f0_hz, p["harmonic_amps"])  # [batch, time]

        # 2. ADSR
        env = self.adsr(p["attack"], p["decay"], p["sustain"], p["release"])
        audio = audio * env

        # 3. Filter
        audio = self.filter_l(audio, p["cutoff"], p["resonance"])

        # 4. Noise
        batch_size = tf.shape(audio)[0]
        noise = tf.random.normal([batch_size, TIME_LENGTH]) * tf.reshape(
            p["noise_amount"], [-1, 1]
        )
        audio = audio + noise

        # 5. tanh
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
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            audio, pitch, timbre_id = inputs
        else:
            audio = inputs
            pitch = tf.zeros([tf.shape(audio)[0]], dtype=tf.float32)
            timbre_id = tf.zeros([tf.shape(audio)[0]], dtype=tf.int32)

        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder([audio, cond], training=training)
        z = sample_z(z_mean, z_logvar)
        p = self.decoder([z, cond], training=training)
        x_hat = self._synthesize_from_params(
            p, self._pitch_to_f0(pitch), training
        )
        return x_hat[:, :, None], z_mean, z_logvar

    # ----------------------------------------------------------
    # generate / generate_blend
    # ----------------------------------------------------------
    def generate(self, pitch, timbre_id, temperature=1.0):
        cond = self._make_cond(pitch, timbre_id)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)
        audio = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
        return audio[:, :, None]

    def generate_blend(self, pitch, timbre_weights, temperature=1.0):
        cond = self.timbre_embedding.blend(pitch, timbre_weights)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)
        audio = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
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
    ) -> DDSPParams:
        """
        DDSPParams を推論して返す。
        GUIはこの結果を受け取り、各パラメータを個別に表示・操作できる。

        Returns:
            DDSPParams インスタンス (GUIで直接操作可能)
        """
        if timbre_weights is not None:
            cond = self.timbre_embedding.blend(pitch, timbre_weights)
        elif timbre_id is not None:
            cond = self._make_cond(pitch, timbre_id)
        else:
            raise ValueError("timbre_id か timbre_weights を指定してください")

        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)

        midi = float(pitch[0]) * 35.0 + 36.0
        f0_hz = float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))

        return DDSPParams(
            f0_hz=f0_hz,
            harmonic_amps=p["harmonic_amps"][0].numpy().tolist(),
            attack=float(p["attack"][0]),
            decay=float(p["decay"][0]),
            sustain=float(p["sustain"][0]),
            release=float(p["release"][0]),
            cutoff=float(p["cutoff"][0]),
            resonance=float(p["resonance"][0]),
            noise_amount=float(p["noise_amount"][0]),
        )

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
        p = self.decoder([z, cond], training=False)
        audio_out = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
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
        (audio, pitch, timbre_id), _ = data

        with tf.GradientTape() as tape:
            cond = self._make_cond(pitch, timbre_id)

            z_mean, z_logvar = self.encoder([audio, cond], training=True)
            z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)
            z = tf.clip_by_value(sample_z(z_mean, z_logvar), -10.0, 10.0)

            p = self.decoder([z, cond], training=True)

            f0_hz = self._pitch_to_f0(pitch)
            x_hat_audio = self._synthesize_from_params(p, f0_hz, training=True)
            x_hat_audio = x_hat_audio[:, :TIME_LENGTH]

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

            # 倍音プロファイル補助損失:
            # timbre_id に対応するテンプレートと harmonic_amps の MSE を計算。
            # ブレンドで生成された場合 (timbre_idが不確か) でも学習データには
            # 常に単一カテゴリのIDが入っているので安全に使える。
            harm_timbre_l = s(
                compute_harmonic_timbre_loss(p["harmonic_amps"], timbre_id)
            )

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
                + harm_timbre_l * 3.0  # 倍音プロファイル補助損失
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
            "harm_timbre": harm_timbre_l,
            "kl": kl,
            "kl_weight": kl_w,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }

    # ----------------------------------------------------------
    # test_step  (val_loss)
    # ----------------------------------------------------------
    def test_step(self, data):
        (audio, pitch, timbre_id), _ = data

        cond = self._make_cond(pitch, timbre_id)

        z_mean, z_logvar = self.encoder([audio, cond], training=False)
        z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)
        z = tf.clip_by_value(sample_z(z_mean, z_logvar), -10.0, 10.0)

        p = self.decoder([z, cond], training=False)

        f0_hz = self._pitch_to_f0(pitch)
        x_hat_audio = self._synthesize_from_params(p, f0_hz, training=False)
        x_hat_audio = x_hat_audio[:, :TIME_LENGTH]

        x_target = tf.clip_by_value(tf.squeeze(audio, axis=-1), -1.0, 1.0)
        x_hat_sq = tf.clip_by_value(x_hat_audio, -1.0, 1.0)

        s = self._safe
        recon = s(tf.reduce_mean(tf.square(x_target - x_hat_sq)))
        stft_l, mel_l, _ = Loss(x_target, x_hat_sq, fft_size=2048, hop_size=512)
        stft_l = s(stft_l)
        mel_l = s(mel_l)
        hf_l = s(compute_high_freq_emphasis_loss(x_target, x_hat_sq))
        flux_l = s(compute_spectral_flux_loss(x_target, x_hat_sq), 0.1)
        harm_timbre_l = s(
            compute_harmonic_timbre_loss(p["harmonic_amps"], timbre_id)
        )

        kl_per_dim = -0.5 * (
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )
        kl_mean = tf.reduce_mean(kl_per_dim, axis=[0, 1])
        kl = s(
            tf.clip_by_value(
                tf.reduce_mean(tf.maximum(kl_mean, self.free_bits)), 0.0, 100.0
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
            + harm_timbre_l * 3.0
            + kl * kl_w
        )
        loss = s(loss, 1000.0)

        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_l,
            "mel": mel_l,
            "high_freq": hf_l,
            "spectral_flux": flux_l,
            "harm_timbre": harm_timbre_l,
            "kl": kl,
            "kl_weight": kl_w,
        }
