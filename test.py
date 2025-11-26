import tensorflow as tf
from train import build_decoder

latent_dim = 128
cond_dim = 4

# z は (B, T, latent_dim)
z = tf.random.normal((1, 10, latent_dim))

# c は (B, cond_dim) で時間次元なし！
c = tf.zeros((1, cond_dim))

decoder = build_decoder()

test = decoder([z, c])
print(tf.reduce_min(test), tf.reduce_max(test))