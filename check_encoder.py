import tensorflow as tf
import numpy as np
from train import WaveCVAE, WaveDecoder, Reparam

custom_objects = {
    "WaveCVAE": WaveCVAE,
    "WaveDecoder": WaveDecoder,
    "Reparam": Reparam,
}

model = tf.keras.models.load_model(
    "checkpoints/wavecvae.model.keras", custom_objects=custom_objects
)

# ランダム波形を入力してエンコーダ出力を観察
x = tf.random.normal((1, 16000, 1))
cond = tf.zeros((1, 4))
mean, logvar = model.encoder([x, cond], training=False)

print("mean:", mean.numpy()[0, :5])      # 最初の5次元だけ表示
print("logvar:", logvar.numpy()[0, :5])
print("mean std:", tf.math.reduce_std(mean).numpy())
print("logvar mean:", tf.reduce_mean(logvar).numpy())
