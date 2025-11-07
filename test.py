from train import WaveCVAE, WaveDecoder, Reparam
import tensorflow as tf
import numpy as np
import soundfile as sf
import librosa

custom_objects = {
    "WaveCVAE": WaveCVAE,
    "WaveDecoder": WaveDecoder,
    "Reparam": Reparam,
}

# -----------------------------
# モデルロード
# -----------------------------
model = tf.keras.models.load_model(
    "checkpoints/wavecvae.model.keras", custom_objects=custom_objects
)
print("✅ モデルをロードしました")

# -----------------------------
# 入力波形と条件
# -----------------------------
y, sr = librosa.load("datasets/input_data/0001.wav", sr=32000)
y = np.expand_dims(y, axis=(0, -1))  # [1, T, 1]
print("入力波形 shape:", y.shape)

cond = np.array([[0.5, 0.5, 0.0, 1.0]], dtype=np.float32)
print("条件ベクトル:", cond)

# -----------------------------
# Encoder 出力とサンプリング
# -----------------------------
mu, logvar = model.encoder([y, cond], training=False)
print("mu:", mu.numpy())
print("logvar:", logvar.numpy())

z = model.reparam([mu, logvar])
print("サンプリング後 z:", z.numpy())

target_time = y.shape[1]
print("target_time:", target_time)

# -----------------------------
# Decoder で波形生成
# -----------------------------
decoder = model.decoder
y_pred = decoder(z, cond, target_time=target_time, training=False)
y_pred = y_pred.numpy().squeeze()

# -----------------------------
# 出力の長さ調整
# -----------------------------
if len(y_pred) < target_time:
    pad = target_time - len(y_pred)
    y_pred = np.pad(y_pred, (0, pad))
else:
    y_pred = y_pred[:target_time]

sf.write("output.wav", y_pred, sr)
print("Decoder 出力 shape:", y_pred.shape)
print("Decoder 出力 min/max/mean:",
      np.min(y_pred), np.max(y_pred), np.mean(y_pred))

output = tf.squeeze(y_pred).numpy()
output /= np.max(np.abs(output) + 1e-9)
sf.write("output_amplified.wav", output, sr)
