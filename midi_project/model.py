import tensorflow as tf
from loss import Loss

SR = 48000
COND_DIM = 3 + 1  # screech, acid, pluck + pitch
LATENT_DIM = 32  # 各時間ステップあたりSR = 48000
WAV_LENGTH = 1.3
TIME_LENGTH = int(WAV_LENGTH * SR)
STFT_FFTS = [256, 512, 1024]  # 頭打ちになるたびに2048,4096を追加
channels = [(64, 7, 2), (128, 5, 2), (256, 5, 2), (256, 3, 2)]


class FiLM(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.gamma = tf.keras.layers.Dense(channels)
        self.beta = tf.keras.layers.Dense(channels)

    def call(self, x, cond):
        # x: (B, T, C)
        g = tf.tanh(self.gamma(cond)[:, None, :])
        b = tf.tanh(self.beta(cond)[:, None, :])
        return x + (g * x + b) * 0.1


def build_encoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1))
    cond = tf.keras.Input(shape=(cond_dim,))

    x = x_in

    for ch, k, s in channels:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = FiLM(ch)(x, cond)
        x = tf.keras.layers.ReLU()(x)

    # latent mean / logvar (time-wise)
    z_mean = tf.keras.layers.Conv1D(latent_dim, 1)(x)
    z_logvar = tf.keras.layers.Conv1D(latent_dim, 1)(x)

    return tf.keras.Model([x_in, cond], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(TIME_LENGTH // 16, latent_dim))
    cond = tf.keras.Input(shape=(cond_dim,))

    x = z_in

    for ch, k, s in reversed(channels):
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(ch, k, padding="same")(x)
        x = FiLM(ch)(x, cond)
        x = tf.keras.layers.ReLU()(x)

    out = tf.keras.layers.Conv1D(1, 1, activation="tanh")(x)
    return tf.keras.Model([z_in, cond], out, name="decoder")


class TimeWiseCVAE(tf.keras.Model):
    def __init__(self, cond_dim=COND_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = build_encoder(cond_dim, latent_dim)
        self.decoder = build_decoder(cond_dim, latent_dim)

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_logvar = self.encoder(
            [x, cond]
        )  # pyright: ignore[reportGeneralTypeIssues]
        z = sample_z(z_mean, z_logvar)
        x_hat = self.decoder([z, cond])
        return x_hat, z_mean, z_logvar

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(
                [x, cond]
            )  # pyright: ignore[reportGeneralTypeIssues]
            z = sample_z(z_mean, z_logvar)
            x_hat = self.decoder([z, cond])
            x_hat = x_hat[:, :TIME_LENGTH, :]
            x = tf.squeeze(x, axis=-1)
            x_hat = tf.squeeze(x_hat, axis=-1)
            recon = tf.reduce_mean(tf.square(x - x_hat))
            kl = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )
            stft_loss, mel_loss, diff_loss = Loss(
                x, x_hat, fft_size=1024, hop_size=256
            )

            loss = (
                recon
                + 0.8 * kl
                # + stft_loss * 0.4
                # + mel_loss * 0.2
                # + diff_loss * 0.1
            )  # ← KL弱め diff_loss弱め

        grads = tape.gradient(loss, self.trainable_variables)
        if self.optimizer:
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables)
            )  # pyright: ignore[reportAttributeAccessIssue]

        return {
            "loss": loss,
            "recon": recon,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "diff_loss": diff_loss,
            "kl": kl,
        }
