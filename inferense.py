# model_infer_dynamic_build.py
import tensorflow as tf
import numpy as np
import librosa
from train import (
    build_encoder, build_decoder, WaveTimeConditionalCVAE, write_wav
)

# ====== 音声読み込み ======
filename = "datasets/input_data/0001.wav"
wav, sr = librosa.load(filename, sr=32000)  # SRは学習時と同じに
wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
T = wav.shape[1]
print(f"✅ 読み込み完了: {filename} (長さ: {T} samples)")

# ====== 条件ベクトル ======
cond = tf.constant([[0.5, 0.8, 0.3, 0.7]], dtype=tf.float32)

# ====== モデル構築 ======
encoder = build_encoder()
decoder = build_decoder()
model = WaveTimeConditionalCVAE(encoder, decoder)

# ⚡ 実データに基づいてshape確定（これがビルド相当）
x_in = tf.constant(wav, dtype=tf.float32)
y_in = tf.zeros_like(x_in)  # ダミー出力（学習時の入力形に合わせる）
lx = tf.constant([T], dtype=tf.int32)
ly = tf.constant([T], dtype=tf.int32)
_ = model((x_in, y_in, cond, lx, ly), training=False)
print("✅ モデルshape確定（実データベース）")

# ====== 重みロード ======
model.load_weights("checkpoints_cvae/cvae_055.weights.h5")
print("✅ 重みロード完了")

# ====== 潜在ベクトル推論 ======
mean, logvar = model.encoder(x_in, training=False)
eps = tf.random.normal(shape=tf.shape(mean))
z = mean + tf.exp(0.5 * logvar) * eps

# ====== 再構成 ======
y_hat = model.decoder([z, cond], training=False)
y_hat = tf.squeeze(y_hat).numpy()

# ====== 保存 ======
write_wav("reconstructed.wav", y_hat)
print("🎵 reconstructed.wav 保存完了！")