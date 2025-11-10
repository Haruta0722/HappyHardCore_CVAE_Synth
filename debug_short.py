# debug_short.py
import tensorflow as tf
import numpy as np
import pandas as pd
from train import build_wave_dataset_from_csv, WaveCVAE# STFT shapes & mask
from train import stft_magnitude, stft_loss_masked  # or import from cvae modulee_mask_from_length  # adapt imports

ds = build_wave_dataset_from_csv("datasets/labels.csv", batch_size=4)  # 小バッチ
for batch in ds.take(1):
    (x_batch, cond_batch, mask_frames), y_batch = batch
    print("x_batch.shape", x_batch.shape, "y_batch.shape", y_batch.shape, "cond.shape", cond_batch.shape, "mask.shape", mask_frames.shape)
    break

model = WaveCVAE()
# build like train does:
input_shape = ((None,) + tuple(x_batch.shape[1:]), (None,) + tuple(cond_batch.shape[1:]), (None,) + tuple(mask_frames.shape[1:]))
model.build(input_shape)

# encoder check
x_in = x_batch  # shape [B, N]
x_in_exp = tf.expand_dims(x_in, -1)
cond_batch = tf.cast(cond_batch, tf.float32)
mu, logvar = model.encoder([x_in_exp, cond_batch], training=False)
print("mu mean,std:", tf.reduce_mean(mu).numpy(), tf.math.reduce_std(mu).numpy())
print("logvar mean,std:", tf.reduce_mean(logvar).numpy(), tf.math.reduce_std(logvar).numpy())

# z check
eps = tf.random.normal(tf.shape(mu))
z = mu + tf.exp(0.5 * logvar) * eps
print("z mean,std:", tf.reduce_mean(z).numpy(), tf.math.reduce_std(z).numpy())

# decoder intermediate check: call decoder and inspect
target_time = tf.shape(y_batch)[1]
y_pred = model.decoder(z, cond_batch, target_time, training=False)  # [B, T, 1]
y_pred = tf.squeeze(y_pred, -1)
print("y_pred shape:", y_pred.shape, "y_pred mean,std:", tf.reduce_mean(y_pred).numpy(), tf.math.reduce_std(y_pred).numpy())


mag_true = stft_magnitude(y_batch)
mag_pred = stft_magnitude(y_pred)
print("mag_true.shape", mag_true.shape, "mag_pred.shape", mag_pred.shape)
print("mask_frames shape", mask_frames.shape)
# compute loss value safely
loss = stft_loss_masked(y_batch, y_pred, mask_frames)
print("stft loss", loss.numpy())
