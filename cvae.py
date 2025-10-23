import tensorflow as tf
from tensorflow.keras import layers, Model


# --- Encoder ---
def build_encoder(input_shape=(None, 128), cond_dim=4, latent_dim=32):
    x_in = layers.Input(shape=input_shape, name="input_spectrogram")
    c_in = layers.Input(shape=(cond_dim,), name="condition")

    # 条件を時間方向にbroadcast
    c_broadcast = tf.tile(tf.expand_dims(c_in, axis=1), [1, tf.shape(x_in)[1], 1])  # type: ignore
    x = layers.Concatenate(axis=-1)([x_in, c_broadcast])

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    mu = layers.Dense(latent_dim)(x)
    log_var = layers.Dense(latent_dim)(x)
    return Model([x_in, c_in], [mu, log_var], name="Encoder")


# --- Sampling Layer ---
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


# --- Decoder ---
def build_decoder(output_shape=(None, 128), cond_dim=4, latent_dim=32):
    z_in = layers.Input(shape=(latent_dim,), name="latent")
    c_in = layers.Input(shape=(cond_dim,), name="condition")

    x = layers.Concatenate()([z_in, c_in])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.RepeatVector(128)(x)  # 時間軸の長さに合わせる
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="sigmoid")(x)
    return Model([z_in, c_in], x, name="Decoder")


# --- CVAE Model ---
class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        x, c = inputs
        mu, log_var = self.encoder([x, c])
        z = self.sampling([mu, log_var])
        x_rec = self.decoder([z, c])
        return x_rec

    def compute_loss(self, x, c):
        mu, log_var = self.encoder([x, c])
        z = self.sampling([mu, log_var])
        x_rec = self.decoder([z, c])

        recon_loss = tf.reduce_mean(tf.abs(x - x_rec))  # MAE
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        return recon_loss + 0.001 * kl_loss
