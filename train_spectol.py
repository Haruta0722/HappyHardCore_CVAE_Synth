# mel_cvae.py
# Requires: tensorflow >= 2.10, librosa
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import soundfile as sf
import tensorflow.keras.layers as layers
import json


# -----------------------
# Hyperparams (tuneable)
# -----------------------
EPOCH = 200
SR = 44100
N_FFT = 2048
HOP = 512
N_MELS = 128
BATCH_SIZE = 4
LATENT_DIM = 64
COND_DIM = 4

STFT_FFTS = [512, 1024]
MEL_BINS = N_MELS

KL_ANNEAL_STEPS = 20000
KL_WEIGHT_MAX = 1.0
FREE_BITS = 0.0

W_MEL = 1.0
W_STFT = 1.0
W_BAND = 1.0

BAND_LOW_HZ = 150.0
BAND_HIGH_HZ = 22050.0

LOG_EPS = float(np.log(1e-6))

# データ CSV パス
LABEL_CSV = "datasets/labels.csv"
BASE_DIR = "."

CHECKPOINT_DIR = "checkpoints"
LAST_WEIGHTS = os.path.join(CHECKPOINT_DIR, "last.weights.h5")
LAST_STATE = os.path.join(CHECKPOINT_DIR, "last_state.json")


# -----------------------
# Utility: mel matrix
# -----------------------
def make_mel_matrices(sr=SR, n_fft=N_FFT, n_mels=N_MELS):

    mel_fb = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2.0
    )

    mel_mat = mel_fb.T.astype(np.float32)
    mel_pinv = np.linalg.pinv(mel_mat)
    return mel_mat, mel_pinv


MEL_MAT_np, MEL_PINV_np = make_mel_matrices(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
MEL_MAT = tf.constant(MEL_MAT_np)
MEL_PINV = tf.constant(MEL_PINV_np)


FREQ_BINS = N_FFT // 2 + 1


# -------------------------
# ユーティリティ
# -------------------------
def load_wav(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)

    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.1
    y = y / rms * target_rms
    return y.astype(np.float32)


def write_wav(path, y, sr=SR):
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)


# --- 前提定義（既存の定数）---
# SR, N_FFT, HOP, N_MELS は既に定義済みとする


# numpy 側で mel を作る関数（py_function 用）
def wav_to_mel_py(wav_np):
    # wav_np: np.ndarray, dtype float32, shape (T,) or (T,1)
    if wav_np is None:
        return np.zeros((0, N_MELS), dtype=np.float32)

    wav_np = np.asarray(wav_np, dtype=np.float32).squeeze()
    if wav_np.ndim == 0:
        # スカラーになってしまっているなら空スペクトログラムを返す（保険）
        return np.zeros((0, N_MELS), dtype=np.float32)

    # librosa.stft: shape (freq_bins, frames)
    S = librosa.stft(
        wav_np, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, center=True
    )
    mag = np.abs(S).T  # -> (frames, freq_bins)

    # mel フィルタ行列（numpy版を事前に作っておくと速い）
    # MEL_MAT_np must be shape (freq_bins, N_MELS)
    mel = np.matmul(mag, MEL_MAT_np)  # (frames, n_mels)
    # log scaling
    mel = np.log(mel + 1e-6).astype(np.float32)

    mel[np.isnan(mel)] = 0.0
    mel[np.isinf(mel)] = 0.0
    return mel


# map 関数で py_function を使うラッパー
def map_to_mel(x_wav, y_wav, cond, y_sample_len):
    # x_wav, y_wav are tf.Tensor 1-D float32 (from from_generator)
    # Use tf.py_function to call your numpy-based wav_to_mel_py
    x_mel = tf.py_function(func=wav_to_mel_py, inp=[x_wav], Tout=tf.float32)
    y_mel = tf.py_function(func=wav_to_mel_py, inp=[y_wav], Tout=tf.float32)

    # py_function removes shape info -> set shapes explicitly
    x_mel.set_shape([None, N_MELS])  # (frames_x, n_mels)
    y_mel.set_shape([None, N_MELS])  # (frames_y, n_mels)

    # compute mel frame count for target (as a plain tf.int32 scalar / tensor)
    y_mel_frames = tf.cast(tf.shape(y_mel)[0], tf.int32)

    # keep condition as float32
    cond = tf.cast(cond, tf.float32)

    # Return order that train_step expects: x_mel, y_waveform, cond, y_mel_frames
    # Note: keep y_wav (raw waveform) for STFT loss. We don't return y_mel here as it's not needed in train_step.
    return x_mel, y_wav, cond, y_mel_frames


# 改訂版 make_dataset_from_csv（抜粋）
def make_dataset_from_csv(
    csv_path, base_dir=BASE_DIR, batch_size=BATCH_SIZE, shuffle=True
):
    df = pd.read_csv(csv_path)

    df["input_path"] = df["input_path"].apply(
        lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p
    )
    df["output_path"] = df["output_path"].apply(
        lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p
    )

    def gen():
        for _, row in df.iterrows():
            x = load_wav(row["input_path"])
            y = load_wav(row["output_path"])
            cond = np.array(
                [
                    row["attack"],
                    row["distortion"],
                    row["thickness"],
                    row["center_tone"],
                ],
                dtype=np.float32,
            )
            yield x, y, cond, np.int32(len(y))

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # x wav
        tf.TensorSpec(shape=(None,), dtype=tf.float32),  # y wav
        tf.TensorSpec(shape=(4,), dtype=tf.float32),  # cond
        tf.TensorSpec(shape=(), dtype=tf.int32),  # y_length
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=256)

    # ① wav → mel
    ds = ds.map(map_to_mel, num_parallel_calls=tf.data.AUTOTUNE)

    # ③ padding
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [None, N_MELS],  # x_mel
            [None],  # y wav
            [4],  # cond
            (),  # y_len
        ),
        padding_values=(
            tf.constant(LOG_EPS, dtype=tf.float32),  # mel の padding（重要）
            tf.constant(0.0, dtype=tf.float32),  # y waveform の padding
            tf.constant(0.0, dtype=tf.float32),  # cond
            tf.constant(0, dtype=tf.int32),  # length
        ),
    )

    ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------
# FiLM layer (time-broadcasting)
# -----------------------
class FiLM(layers.Layer):
    def __init__(self, channels, cond_dim=COND_DIM, hidden=128, name=None):
        super().__init__(name=name)
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        self.net = tf.keras.Sequential(
            [
                layers.Dense(hidden, activation="relu"),
                layers.Dense(hidden, activation="relu"),
                layers.Dense(
                    self.channels * 2,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
            ]
        )

    def call(self, inputs):
        """
        inputs:
          x: [B, T, C]
          cond: [B, cond_dim] or [B, T, cond_dim]
        output: x * (1+gamma) + beta
        """
        x, cond = inputs
        # cond -> [B, 2*C]
        gb = self.net(cond)
        gamma = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        # expand time axis: [B,1,C]
        gamma = tf.expand_dims(gamma, axis=1)
        beta = tf.expand_dims(beta, axis=1)
        return x * (1.0 + gamma) + beta


# -----------------------
# Encoder: Conv1D-based, keeps time axis
# input: mel [B, T, n_mels]
# output: mean & logvar [B, T, latent_dim]
# -----------------------
def build_encoder(latent_dim=LATENT_DIM, chs=[128, 128, 128], kernel=3):
    inp = layers.Input(shape=(None, N_MELS), name="enc_mel_in")
    x = inp
    for i, c in enumerate(chs):
        x = layers.Conv1D(
            c, kernel_size=kernel, padding="same", name=f"enc_conv_{i}"
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

    # optionally a few dilated blocks (keeps time)
    for i, d in enumerate([1, 2, 4]):
        h = layers.Conv1D(
            chs[-1],
            kernel_size=3,
            dilation_rate=d,
            padding="same",
            name=f"enc_dil_{i}",
        )(x)
        h = layers.Activation("relu")(h)
        x = layers.Add(name=f"enc_res_{i}")([x, h])

    mean = layers.Conv1D(
        latent_dim, kernel_size=1, padding="same", name="enc_mean"
    )(x)
    logvar = layers.Conv1D(
        latent_dim, kernel_size=1, padding="same", name="enc_logvar"
    )(x)

    return tf.keras.Model(inp, [mean, logvar], name="time_encoder")


# -----------------------
# Reparam (time-distributed)
# -----------------------
class ReparamTime(layers.Layer):
    def call(self, mean, logvar, training=True):
        if training:
            eps = tf.random.normal(tf.shape(mean))
            return mean + tf.exp(0.5 * logvar) * eps
        else:
            return mean


# -----------------------
# gated residual block
# -----------------------
def gated_residual_block(
    x,
    channels,
    dilation=1,
    name=None,
    cond=None,
    use_film_inside=False,
    film_hidden=128,
):
    """Keras functional safe version. cond is [B,cond_dim] or [B,T,cond_dim]"""
    h = layers.Conv1D(
        filters=channels * 2,
        kernel_size=3,
        dilation_rate=dilation,
        padding="same",
        name=f"{name}_conv",
    )(x)
    # split
    a = layers.Lambda(lambda t: t[:, :, :channels], name=f"{name}_split_a")(h)
    b = layers.Lambda(lambda t: t[:, :, channels:], name=f"{name}_split_b")(h)

    if cond is not None and use_film_inside:
        # produce gb [B, 2C] or [B,T,2C]
        cond_rank = len(cond.shape)
        if cond_rank == 2:
            gb = layers.Dense(
                film_hidden, activation="relu", name=f"{name}_film1"
            )(cond)
            gb = layers.Dense(
                channels * 2,
                kernel_initializer="zeros",
                bias_initializer="zeros",
                name=f"{name}_film2",
            )(gb)
            gb = layers.Lambda(
                lambda t: tf.expand_dims(t, 1), name=f"{name}_film_expand"
            )(gb)
        else:
            gb = layers.TimeDistributed(
                layers.Dense(
                    channels * 2,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
                name=f"{name}_film_td",
            )(cond)

        gamma = layers.Lambda(
            lambda t: t[..., :channels], name=f"{name}_gamma"
        )(gb)
        beta = layers.Lambda(lambda t: t[..., channels:], name=f"{name}_beta")(
            gb
        )
        gamma_p1 = layers.Lambda(lambda g: g + 1.0, name=f"{name}_gamma_p1")(
            gamma
        )
        a = layers.Multiply(name=f"{name}_film_mul")([a, gamma_p1])
        a = layers.Add(name=f"{name}_film_add")([a, beta])

    a = layers.Activation("tanh", name=f"{name}_tanh")(a)
    b = layers.Activation("sigmoid", name=f"{name}_sigmoid")(b)
    g = layers.Multiply(name=f"{name}_gate")([a, b])

    proj = layers.Conv1D(
        filters=channels,
        kernel_size=1,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_proj",
    )(g)
    proj = layers.Lambda(lambda t: t * 0.1, name=f"{name}_scale")(proj)

    # ensure input channels
    in_ch = tf.keras.backend.int_shape(x)[-1]
    if in_ch is None or in_ch != channels:
        x_res = layers.Conv1D(
            filters=channels,
            kernel_size=1,
            padding="same",
            name=f"{name}_inproj",
        )(x)
    else:
        x_res = x

    out = layers.Add(name=f"{name}_out")([x_res, proj])
    return out


# -----------------------
# Decoder
# -----------------------
def build_decoder(
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    channels=[128, 128, 64],
    upsample_factors=[1, 1, 1],
    n_dilated_per_stage=3,
    use_film_in_residual=True,
):

    z_in = layers.Input(shape=(None, latent_dim), name="dec_z")
    cond_in = layers.Input(shape=(cond_dim,), name="dec_cond")

    x = layers.Conv1D(
        channels[0], kernel_size=3, padding="same", name="dec_proj"
    )(z_in)
    # FiLM (global cond broadcast)
    x = FiLM(channels[0], cond_dim=cond_dim, name="dec_proj_film")([x, cond_in])
    x = layers.Activation("relu")(x)

    for stage_idx, ch in enumerate(channels):

        if stage_idx != 0:
            x = layers.Conv1D(
                ch, kernel_size=3, padding="same", name=f"dec_conv_{stage_idx}"
            )(x)
            x = layers.Activation("relu")(x)

        for i in range(n_dilated_per_stage):
            d = 2 ** (i % 4)
            x = gated_residual_block(
                x,
                channels=ch,
                dilation=d,
                name=f"dec_stage{stage_idx}_res{i}",
                cond=cond_in if use_film_in_residual else None,
                use_film_inside=use_film_in_residual,
            )

        x = FiLM(ch, cond_dim=cond_dim, name=f"dec_stage{stage_idx}_film")(
            [x, cond_in]
        )
        x = layers.Activation("relu")(x)

    out = layers.Conv1D(
        N_MELS, kernel_size=1, padding="same", activation=None, name="dec_out"
    )(x)

    return tf.keras.Model([z_in, cond_in], out, name="mel_decoder")


# -----------------------
# Loss utilities
# -----------------------
def stft_mag_and_phase(wave, n_fft, hop):
    S = tf.signal.stft(
        wave, frame_length=n_fft, frame_step=hop, fft_length=n_fft
    )
    mag = tf.abs(S)
    phase = tf.math.angle(S)
    return mag, phase


def spectral_convergence_from_mag(S_mag, S_hat_mag, eps=1e-7):
    num = tf.sqrt(tf.reduce_sum(tf.square(S_mag - S_hat_mag), axis=[1, 2]))
    den = tf.sqrt(tf.reduce_sum(tf.square(S_mag), axis=[1, 2]))
    return tf.reduce_mean(num / (den + eps))


def magnitude_l1_from_mag(S_mag, S_hat_mag):
    return tf.reduce_mean(tf.abs(S_mag - S_hat_mag))


def log_mag_l1_from_mag(S_mag, S_hat_mag, eps=1e-7):
    return tf.reduce_mean(
        tf.abs(tf.math.log(S_mag + eps) - tf.math.log(S_hat_mag + eps))
    )


def mel_l1_from_mel_log(mel_log, mel_hat_log):
    # both are in log-domain already: [B, T, n_mels]
    # mask 算出は呼び出し側で行う（既にしているなら不要）
    return tf.reduce_mean(tf.abs(mel_log - mel_hat_log))


def band_out_penalty_from_mel_linear(mel_hat_linear, sr=SR):
    # mel_hat_linear: linear mel magnitudes [B, T, n_mels]
    mel_f = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=sr / 2.0)
    mask = np.logical_or(mel_f < BAND_LOW_HZ, mel_f > BAND_HIGH_HZ).astype(
        np.float32
    )
    mask_tf = tf.constant(mask[None, None, :])
    penalty = tf.reduce_mean(tf.abs(mel_hat_linear * mask_tf))
    return penalty


def mel_to_linear_mag(mel_hat):

    S_hat_mag = tf.linalg.matmul(mel_hat, MEL_PINV)

    S_hat_mag = tf.maximum(S_hat_mag, 1e-7)
    return S_hat_mag


def waveform_to_linear_mag(y, n_fft, hop):

    S = tf.signal.stft(y, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    S_mag = tf.abs(S)
    S_mag = tf.maximum(S_mag, 1e-7)
    return S_mag


def waveform_to_mel(y, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    S_mag = waveform_to_linear_mag(y, n_fft=n_fft, hop=hop)

    mel = tf.linalg.matmul(S_mag, MEL_MAT)
    mel = tf.maximum(mel, 1e-7)
    return mel


# -----------------------
# CVAE Model (custom train_step)
# -----------------------
class MelTimeCVAE(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim=LATENT_DIM,
        kl_anneal_steps=KL_ANNEAL_STEPS,
        kl_weight_max=KL_WEIGHT_MAX,
        free_bits=FREE_BITS,
        stft_w=W_STFT,
        mel_w=W_MEL,
        band_w=W_BAND,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparam = ReparamTime()
        self.latent_dim = latent_dim
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_weight_max = kl_weight_max
        self.free_bits = free_bits
        self.stft_w = stft_w
        self.mel_w = mel_w
        self.band_w = band_w

        self.total_steps = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="total_steps"
        )

        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_mel_tracker = tf.keras.metrics.Mean(name="mel_loss")
        self.recon_stft_tracker = tf.keras.metrics.Mean(name="stft_loss")
        self.band_tracker = tf.keras.metrics.Mean(name="band_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")
        self.mean_logvar_tracker = tf.keras.metrics.Mean(name="mean_logvar")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.recon_mel_tracker,
            self.recon_stft_tracker,
            self.band_tracker,
            self.kl_tracker,
            self.mean_logvar_tracker,
        ]

    def kl_weight(self):
        step = tf.cast(self.total_steps, tf.float32)
        s = tf.minimum(1.0, step / tf.cast(self.kl_anneal_steps, tf.float32))
        return s * self.kl_weight_max

    def train_step(self, data):

        x_mel, y, cond, len_frames = data
        if len_frames is not None:
            max_frames = tf.shape(x_mel)[1]
            mask = tf.cast(
                tf.sequence_mask(len_frames, maxlen=max_frames), tf.float32
            )
            mask = tf.expand_dims(mask, -1)  # [B, T, 1]
            x_mel_masked = x_mel * mask
        else:
            x_mel_masked = x_mel
            mask = None

        with tf.GradientTape() as tape:

            # マスク適用後の x_mel, mel_hat
            # tf.print("x_mel_masked min/max:", tf.reduce_min(x_mel_masked), tf.reduce_max(x_mel_masked))
            # tf.debugging.check_numerics(x_mel_masked, "x_mel")

            mean, logvar = self.encoder(x_mel_masked, training=True)
            # tf.print("mean logvar stats:", tf.reduce_min(mean), tf.reduce_max(mean), tf.reduce_min(logvar), tf.reduce_max(logvar))
            # tf.debugging.check_numerics(mean, "mean")
            # tf.debugging.check_numerics(logvar, "logvar")
            z = self.reparam(mean, logvar, training=True)

            mel_hat = self.decoder([z, cond], training=True)
            if len_frames is not None:
                max_frames = tf.shape(x_mel)[1]
                mask = tf.cast(
                    tf.sequence_mask(len_frames, maxlen=max_frames), tf.float32
                )
                mask = tf.expand_dims(mask, -1)  # [B, T, 1]
                mel_hat = mel_hat * mask
            else:
                pass
            # tf.print("mel_hat min/max:", tf.reduce_min(mel_hat), tf.reduce_max(mel_hat))
            # tf.debugging.check_numerics(mel_hat, "mel_hat")

            # 再構成損失（log-domain 同士の L1）
            mel_loss = mel_l1_from_mel_log(x_mel_masked, mel_hat)

            # mel_hat を linear mel に戻す（数値安定化）
            eps_small = 1e-6
            mel_hat_linear = tf.exp(
                mel_hat
            )  # since x_mel = log(linear + eps) originally
            mel_hat_linear = tf.maximum(mel_hat_linear, 1e-7)

            # それを線形スペクトルに戻す（既存関数を利用）
            S_hat_mag = mel_to_linear_mag(
                mel_hat_linear
            )  # expects linear mel -> uses MEL_PINV
            S_mag_target = waveform_to_linear_mag(y, n_fft=N_FFT, hop=HOP)

            # フレーム数合わせて STFT 損失
            min_frames = tf.minimum(
                tf.shape(S_hat_mag)[1], tf.shape(S_mag_target)[1]
            )
            S_hat_c = S_hat_mag[:, :min_frames, :]
            S_tar_c = S_mag_target[:, :min_frames, :]

            sc = spectral_convergence_from_mag(S_tar_c, S_hat_c)
            mag = magnitude_l1_from_mag(S_tar_c, S_hat_c)
            logm = log_mag_l1_from_mag(S_tar_c, S_hat_c)
            stft_loss = sc + mag + logm

            band_loss = band_out_penalty_from_mel_linear(mel_hat_linear)

            kl_per = -0.5 * (1.0 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_time = tf.reduce_sum(kl_per, axis=-1)

            free_total = self.free_bits * tf.cast(self.latent_dim, tf.float32)
            kl_time_clipped = tf.maximum(kl_time, free_total)
            kl_loss = tf.reduce_mean(kl_time_clipped)

            kl_w = self.kl_weight()

            loss = (
                self.mel_w * mel_loss
                + self.stft_w * stft_loss
                + self.band_w * band_loss
                + kl_w * kl_loss
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_steps.assign_add(1)
        self.loss_tracker.update_state(loss)
        self.recon_mel_tracker.update_state(mel_loss)
        self.recon_stft_tracker.update_state(stft_loss)
        self.band_tracker.update_state(band_loss)
        self.kl_tracker.update_state(kl_loss)
        self.mean_logvar_tracker.update_state(tf.reduce_mean(logvar))

        return {
            "loss": self.loss_tracker.result(),
            "mel_loss": self.recon_mel_tracker.result(),
            "stft_loss": self.recon_stft_tracker.result(),
            "band_loss": self.band_tracker.result(),
            "kl": self.kl_tracker.result(),
            "kl_w": kl_w,
            "mean_logvar": self.mean_logvar_tracker.result(),
            "step": self.total_steps,
        }

    def call(self, inputs, training=False):
        # inputs は [mel, cond] のような形式で受け取る
        mel, cond = inputs

        print("mel:", mel.shape)
        print("cond:", cond.shape)
        # 1) エンコード
        z_mean, z_logvar = self.encoder(mel, training=training)

        # 2) サンプリング
        z = self.reparam(z_mean, z_logvar)

        # 3) デコード
        recon = self.decoder([z, cond], training=training)

        # 4) 必要なら KL も返す
        return recon, z_mean, z_logvar


def make_model():
    enc = build_encoder(latent_dim=LATENT_DIM, chs=[128, 128, 128])
    dec = build_decoder(
        latent_dim=LATENT_DIM,
        cond_dim=COND_DIM,
        channels=[128, 128, 64],
        n_dilated_per_stage=3,
        use_film_in_residual=True,
    )
    model = MelTimeCVAE(enc, dec)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    return model


def save_state(epoch):
    state = {"epoch": epoch}
    with open(LAST_STATE, "w") as f:
        json.dump(state, f)


def load_state():
    if not os.path.exists(LAST_STATE):
        return 0
    with open(LAST_STATE, "r") as f:
        state = json.load(f)
    return int(state.get("epoch", 0))


def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------------------------
    # データセット
    # ---------------------------
    ds = make_dataset_from_csv(
        LABEL_CSV, base_dir=".", batch_size=4, shuffle=True
    )
    # ---------------------------
    # モデル
    # ---------------------------
    model = make_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    sample_batch = next(iter(ds.take(1)))

    x_mel_s, y_s, cond_s, frames_s = sample_batch

    _ = model([x_mel_s, cond_s])  # build the model (encoder->decoder) once

    # ---------------------------
    # 途中再開
    # ---------------------------
    initial_epoch = load_state()

    if os.path.exists(LAST_WEIGHTS):
        print(f"Resuming from epoch {initial_epoch}, loading {LAST_WEIGHTS}")
        model.load_weights(LAST_WEIGHTS)
    else:
        print("No checkpoint found — training from scratch")

    # ---------------------------
    # チェックポイント（エポックごと）
    # ---------------------------
    epoch_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CHECKPOINT_DIR, "mel_cvae_epoch_{epoch:03d}.weights.h5"
        ),
        save_weights_only=True,
        save_freq="epoch",
    )

    # 最新重み保存
    last_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=LAST_WEIGHTS, save_weights_only=True, save_freq="epoch"
    )

    # ---------------------------
    # epoch 数を保存するコールバック
    # ---------------------------
    class StateCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_state(epoch + 1)  # 次の epoch から開始できるよう +1

    state_cb = StateCallback()

    # ---------------------------
    # 学習率制御
    # ---------------------------
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=5, min_lr=1e-6
    )

    # ---------------------------
    # 学習開始
    # ---------------------------
    model.fit(
        ds,
        epochs=EPOCH,
        initial_epoch=initial_epoch,
        steps_per_epoch=243,
        callbacks=[epoch_ckpt_cb, last_ckpt_cb, state_cb, lr_cb],
    )


if __name__ == "__main__":
    main()
