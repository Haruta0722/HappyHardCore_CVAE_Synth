import tensorflow as tf


def stft_loss(x, x_rec, frame_length=512, frame_step=256):
    stft_x = tf.signal.stft(x, frame_length, frame_step)
    stft_x_rec = tf.signal.stft(x_rec, frame_length, frame_step)
    mag_x = tf.abs(stft_x)
    mag_x_rec = tf.abs(stft_x_rec)
    return tf.reduce_mean(tf.abs(mag_x - mag_x_rec))


def total_loss(x, x_rec, mu, log_var, alpha=1.0, gamma=0.001):
    stft = stft_loss(x, x_rec)
    kl = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return alpha * stft + gamma * kl
