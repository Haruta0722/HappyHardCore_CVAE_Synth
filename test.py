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

# モデル丸ごとロード
model = tf.keras.models.load_model(
    "checkpoints/wavecvae.keras", custom_objects=custom_objects
)

# ダミー入力で動作確認

dummy_audio = tf.zeros((1, 16000, 1), dtype=tf.float32)
dummy_cond = tf.zeros((1, 4), dtype=tf.float32)
output = model([dummy_audio, dummy_cond], training=False)
print("出力 shape:", output.shape)
print("出力 mean:", tf.reduce_mean(output).numpy())

y, sr = librosa.load("datasets/input_data/0001.wav", sr=32000)
y = np.expand_dims(y, axis=(0, -1))  # [1, T, 1]

cond = np.array([[0.5, 0.5, 0.0, 1.0]], dtype=np.float32)

# 推論
output = model([y, cond], training=False)
output = tf.squeeze(output).numpy()
print("min:", output.min(), "max:", output.max(), "mean:", output.mean())

sf.write("output.wav", output, sr)
