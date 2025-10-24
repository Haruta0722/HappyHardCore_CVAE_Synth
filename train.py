# cvae_waveform_stft.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import librosa

# ---------------------
# ハイパーパラメータ
# ---------------------
SR = 32000
FRAME_LENGTH = 1024
FRAME_STEP = 256

LATENT_DIM = 64
COND_DIM = 4
BATCH_SIZE = 4
EPOCHS = 50
KL_WEIGHT = 1e-3
LR = 1e-4


# ---------------------
# データ読み込み -> waveform（任意長）関数
# ---------------------
def load_wav(path: tf.Tensor | str) -> tf.Tensor:
    if isinstance(path, tf.Tensor):
        path_str = path.numpy().decode("utf-8")
    else:
        path_str = str(path)
    audio, sr = librosa.load(path_str, sr=32000, mono=True)
    return tf.convert_to_tensor(audio, dtype=tf.float32)


def make_mask_from_length(lengths, max_len):
    """lengths: [B] int32 tensor (samples), max_len: int -> produce boolean mask per sample in frames"""
    # convert sample-lengths to frame counts used by STFT:
    # number of frames = floor((n_samples - frame_length) / frame_step) + 1 for n_samples >= frame_length
    # We'll compute frames = tf.maximum(0, (n_samples - FRAME_LENGTH) // FRAME_STEP + 1)
    frames = tf.maximum(0, (lengths - FRAME_LENGTH) // FRAME_STEP + 1)
    # But batch will be padded to max_samples; compute max_frames accordingly
    max_frames = tf.maximum(0, (max_len - FRAME_LENGTH) // FRAME_STEP + 1)
    # create mask [B, max_frames]
    rng = tf.range(max_frames)
    frames = tf.expand_dims(frames, axis=1)  # [B,1]
    mask = tf.less(
        rng, frames
    )  # broadcasting -> [max_frames] < [B,1] -> [B, max_frames]
    mask = tf.cast(mask, tf.bool)
    return mask  # [B, T_frames]


# ---------------------
# STFT loss (masked)
# ---------------------
def stft_magnitude(x):
    # x: [B, N] waveform (padded)
    stfts = tf.signal.stft(
        x,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
        window_fn=tf.signal.hann_window,
    )
    mag = tf.abs(stfts)  # [B, T_frames, F_bins]
    return mag


def stft_loss_masked(y_true_wav, y_pred_wav, mask_frames):
    """
    y_true_wav, y_pred_wav: [B, N_samples] (padded)
    mask_frames: [B, T_frames] boolean -> True for valid frames
    """
    mag_true = stft_magnitude(y_true_wav)
    mag_pred = stft_magnitude(y_pred_wav)
    # log scaling helps stability
    log_true = tf.math.log(mag_true + 1e-7)
    log_pred = tf.math.log(mag_pred + 1e-7)
    # expand mask to freq axis
    mask_f = tf.cast(
        tf.expand_dims(mask_frames, axis=-1), tf.float32
    )  # [B, T, 1]
    diff = tf.abs(log_true - log_pred) * mask_f
    # average only over valid frames and freq bins
    denom = tf.reduce_sum(mask_f) * tf.cast(tf.shape(log_true)[-1], tf.float32)
    loss = tf.reduce_sum(diff) / (denom + 1e-8)
    return loss


# ---------------------
# CVAE (waveform -> waveform)
# ---------------------
def build_wave_encoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    """
    Encoder works on [B, T, 1] waveform (padded)
    """
    x_in = layers.Input(shape=(None, 1), name="wave_in")  # time-dim dynamic
    c_in = layers.Input(shape=(cond_dim,), name="cond")

    # broadcast cond across time
    def concat_cond(inputs):
        x, c = inputs
        t = tf.shape(x)[1]
        c_t = tf.expand_dims(c, axis=1)  # [B,1,C]
        c_b = tf.tile(c_t, [1, t, 1])  # [B, t, C]
        return tf.concat([x, c_b], axis=-1)

    h = layers.Lambda(concat_cond)([x_in, c_in])  # [B, T, 1+cond_dim]
    # 1D conv stack with downsampling
    h = layers.Conv1D(64, 9, strides=2, padding="same", activation="relu")(
        h
    )  # /2
    h = layers.Conv1D(128, 9, strides=2, padding="same", activation="relu")(
        h
    )  # /4
    h = layers.Conv1D(256, 9, strides=2, padding="same", activation="relu")(
        h
    )  # /8
    h = layers.GlobalAveragePooling1D()(h)  # [B, channels]
    mu = layers.Dense(latent_dim, name="mu")(h)
    logvar = layers.Dense(latent_dim, name="logvar")(h)
    return Model([x_in, c_in], [mu, logvar], name="wave_encoder")


class Reparam(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps


class WaveDecoder(layers.Layer):
    def __init__(
        self,
        cond_dim=COND_DIM,
        latent_dim=LATENT_DIM,
        upsample_factors=(2, 2, 2),
        channel=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.channel = channel
        # small dense stack to expand
        self.d1 = layers.Dense(1024, activation="relu")
        self.d2 = layers.Dense(self.channel, activation="relu")  # intermediate
        # will expand into time-steps via tile
        self.lstm = layers.Conv1D(
            self.channel, 3, padding="same", activation="relu"
        )  # applied after upsampling
        self.ups = [
            layers.UpSampling1D(size=2),
            layers.UpSampling1D(size=2),
            layers.UpSampling1D(size=2),
        ]
        self.conv_up = layers.Conv1D(
            self.channel, 9, padding="same", activation="relu"
        )
        self.out_conv = layers.Conv1D(
            1, 7, padding="same", activation=None
        )  # raw waveform (no activation)

    def call(self, z, cond, target_time, training=False):
        # z: [B, zdim], cond: [B, cdim], target_time: scalar
        h = tf.concat([z, cond], axis=-1)
        h = self.d1(h)  # [B,1024]
        h = self.d2(h)  # [B,256]
        # create a short seed sequence then upsample to target_time
        seed_len = tf.maximum(1, target_time // (2 ** len(self.ups)))
        h = tf.expand_dims(h, axis=1)  # [B,1,C]
        h = tf.tile(h, [1, seed_len, 1])  # [B, seed_len, C]
        # progressively upsample until reaching >= target_time
        for up in self.ups:
            h = up(h)
            h = self.conv_up(h)
        # now h length should be >= target_time; trim or pad
        cur_len = tf.shape(h)[1]

        def trim():
            return h[:, :target_time, :]

        def pad():
            pad_len = target_time - cur_len
            pad_tensor = tf.zeros(
                [tf.shape(h)[0], pad_len, tf.shape(h)[2]], dtype=h.dtype
            )
            return tf.concat([h, pad_tensor], axis=1)

        h = tf.cond(cur_len >= target_time, trim, pad)
        out = self.out_conv(h)  # [B, target_time, 1]
        return out


# ---------------------
# Full model wrapper with loss computation
# ---------------------
class WaveCVAE(Model):
    def __init__(
        self, latent_dim=LATENT_DIM, cond_dim=COND_DIM, kl_weight=KL_WEIGHT
    ):
        super().__init__()
        self.encoder = build_wave_encoder(
            cond_dim=cond_dim, latent_dim=latent_dim
        )
        self.reparam = Reparam()
        self.decoder = WaveDecoder(cond_dim=cond_dim, latent_dim=latent_dim)
        self.kl_weight = kl_weight
        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

    def call(self, inputs, training=False):
        x, cond = inputs  # x: [B, T, 1]
        mu, logvar = self.encoder([x, cond], training=training)
        z = self.reparam([mu, logvar])
        target_time = tf.shape(x)[1]
        out = self.decoder(z, cond, target_time, training=training)
        return out

    def compute_losses(self, x, cond, mask_frames, y):
        # x, y: [B, N_samples] -> but we assume they are already batched and padded as [B, N]
        # our encoder expects shape [B, T, 1], so expand dims
        x_in = tf.expand_dims(x, axis=-1)  # [B, N, 1]
        mu, logvar = self.encoder([x_in, cond], training=True)
        z = self.reparam([mu, logvar])
        target_time = tf.shape(y)[1]
        y_pred = self.decoder(z, cond, target_time, training=True)  # [B, Ty, 1]
        # squeeze for stft
        y_pred_wav = tf.squeeze(y_pred, axis=-1)  # [B, Ty]
        # ensure y already has shape [B, Ty]
        recon = stft_loss_masked(y, y_pred_wav, mask_frames)
        # KL
        kl_per = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        kl = tf.reduce_mean(tf.reduce_sum(kl_per, axis=1))
        total = recon + self.kl_weight * kl
        return total, recon, kl, y_pred_wav


# ---------------------
# Dataset builder from your CSV
# ---------------------
def build_wave_dataset_from_csv(csv_path, batch_size=BATCH_SIZE):
    df = pd.read_csv(csv_path)
    in_paths = df["input_path"].astype(str).values
    out_paths = df["output_path"].astype(str).values
    attacks = df["attack"].astype(np.float32).values
    distortions = df["distortion"].astype(np.float32).values
    thicknesses = df["thickness"].astype(np.float32).values
    centers = df["center_tone"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices(
        (in_paths, out_paths, attacks, distortions, thicknesses, centers)
    )

    def _py_load(inp, outp, a, d, t, c):
        x = load_wav(inp)
        y = load_wav(outp)
        return x, y, a, d, t, c

    ds = ds.map(
        lambda i, o, a, d, t, c: tf.py_function(
            func=_py_load,
            inp=[i, o, a, d, t, c],
            Tout=[
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
            ],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # pad waveforms to batch-max length (samples)
    def to_padded_example(x, y, a, d, t, c):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        cond = tf.stack([a, d, t, c])
        # return sample-level (x, y, cond) ; actual padding handled by padded_batch
        return x, y, cond

    ds = ds.map(to_padded_example, num_parallel_calls=tf.data.AUTOTUNE)

    # padded_batch: pad waveforms to same sample length within each batch
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([None], [None], [COND_DIM]),
        padding_values=(0.0, 0.0, 0.0),
    )
    ds = ds.map(
        lambda x, y, cond: _post_batch_map(x, y, cond),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _post_batch_map(x_batch, y_batch, cond_batch):
    """
    After padded_batch: x_batch, y_batch are [B, N_samples]
    We need to produce mask_frames for y_batch and lengths if needed.
    """
    nonzero = tf.cast(tf.greater(tf.abs(y_batch), 1e-7), tf.int32)
    lengths = tf.reduce_sum(nonzero, axis=1)  # [B]
    max_len = tf.shape(y_batch)[1]
    mask_frames = make_mask_from_length(lengths, max_len)  # [B, T_frames]
    return (x_batch, cond_batch, mask_frames), y_batch


# ---------------------
# Training loop
# ---------------------
def train(csv_path):
    ds = build_wave_dataset_from_csv(csv_path, batch_size=BATCH_SIZE)
    model = WaveCVAE(
        latent_dim=LATENT_DIM, cond_dim=COND_DIM, kl_weight=KL_WEIGHT
    )
    opt = tf.keras.optimizers.Adam(LR)

    @tf.function
    def step(batch):
        (x_batch, cond_batch, mask_frames), y_batch = batch
        with tf.GradientTape() as tape:
            total, recon, kl, y_pred = model.compute_losses(
                x_batch, cond_batch, mask_frames, y_batch
            )
        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        model.loss_tracker.update_state(total)
        model.recon_tracker.update_state(recon)
        model.kl_tracker.update_state(kl)
        return total, recon, kl

    for epoch in range(EPOCHS):
        for step_i, batch in enumerate(ds):
            total, recon, kl = step(batch)
            if step_i % 10 == 0:
                tf.print(
                    "Epoch",
                    epoch + 1,
                    "Step",
                    step_i,
                    "loss",
                    total,
                    "recon",
                    recon,
                    "kl",
                    kl,
                )
        tf.print("Epoch", epoch + 1, "avg loss", model.loss_tracker.result())
        # reset metrics
        model.loss_tracker.reset_states()
        model.recon_tracker.reset_states()
        model.kl_tracker.reset_states()

    return model


if __name__ == "__main__":
    model = train("datasets/labels.csv")
    save_path = "checkpoints/wavecvae_weights.h5"
    model.save_weights(save_path)
    print(f"✅ モデル重みを保存しました: {save_path}")
