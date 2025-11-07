import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import librosa

# --- すでに学習済みのモデルをロード ---
from train import WaveCVAE, build_wave_dataset_from_csv, COND_DIM, LATENT_DIM

CHECKPOINT_PATH = "checkpoints/wavecvae.weights.h5"
CSV_PATH = "datasets/labels.csv"

# モデル構築・重み読み込み
model = WaveCVAE(latent_dim=LATENT_DIM, cond_dim=COND_DIM)
_ = model(tf.zeros((1, 16000, 1)), tf.zeros((1, COND_DIM)))  # ダミー呼び出しでbuild
model.load_weights(CHECKPOINT_PATH)
print("✅ モデル重みを読み込みました")

# --- データセット準備（条件ベクトルも含める） ---
ds = build_wave_dataset_from_csv(CSV_PATH, batch_size=1)

mus = []
conds = []

for (x, cond, mask), y in ds.take(200):  # 最初の200サンプルのみ
    mu, logvar = model.encoder([x[..., None], cond], training=False)
    mus.append(mu.numpy()[0])
    conds.append(cond.numpy()[0])

mus = np.array(mus)
conds = np.array(conds)

print("潜在ベクトル shape:", mus.shape)  # (N, latent_dim)
print("条件ベクトル shape:", conds.shape)

# --- 次元削減 ---
if mus.shape[1] > 2:
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    mus_2d = reducer.fit_transform(mus)
else:
    mus_2d = mus

# --- 可視化 ---
plt.figure(figsize=(8, 6))
plt.scatter(mus_2d[:, 0], mus_2d[:, 1], c=conds[:, 0], cmap="viridis", alpha=0.7)
plt.colorbar(label="attack 値（例）")
plt.title("Latent Space Visualization (μ) by attack condition")
plt.xlabel("latent dim 1")
plt.ylabel("latent dim 2")
plt.show()