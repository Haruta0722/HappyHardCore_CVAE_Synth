import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import os

from train import WaveCVAE, WaveDecoder,build_wave_encoder 
# ↑ モジュールパスは train.py の構成に合わせて適宜修正

SR = 32000  # サンプリングレート
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "reconstructed"
LATENT_DIM = 64  # 潜在変数の次元数
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("スクリプト実行")
# ==============
# 1. モデル読み込み
# ==============
model = WaveCVAE()
checkpoint = tf.train.Checkpoint(model=model)
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt:
    checkpoint.restore(latest_ckpt).expect_partial()
    print(f"✅ チェックポイントをロードしました: {latest_ckpt}")
    for w in model.weights[:10]:
        print(w.name, tf.reduce_mean(w).numpy(), tf.math.reduce_std(w).numpy())
else:
    raise FileNotFoundError("❌ チェックポイントが見つかりません。")

# ==============
# 2. テスト波形の読み込み
# ==============
test_path = "datasets/input_data/0001.wav"  # ここに任意のテスト音声を配置
if not os.path.exists(test_path):
    raise FileNotFoundError("❌ test.wav が見つかりません。datasets/ に配置してください。")

x, _ = librosa.load(test_path, sr=SR)
x = np.expand_dims(x, axis=(0, -1))  # [1, T, 1] に変形

# 条件ベクトル（もし条件付きなら置き換え）
cond = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
# ==============
# 3. 再構成処理
# ==============
z_rand = tf.random.normal(shape=(1, LATENT_DIM))
cond = np.zeros((1, 4), dtype=np.float32)
x_recon = model.decoder(z_rand, cond, target_time=x.shape[1], training=False)
x_recon = x_recon.numpy().squeeze()
print("rand z std:", np.std(x_recon), "mean:", np.mean(x_recon))
sf.write("reconstructed/z_rand.wav", x_recon, SR)


# 長さを入力に合わせる
min_len = min(len(x_recon), x.shape[1])
x_recon = x_recon[:min_len]
x_in = x.squeeze()[:min_len]

# ==============
# 4. 保存とテキスト出力
# ==============
sf.write(os.path.join(OUTPUT_DIR, "input.wav"), x_in, SR)
sf.write(os.path.join(OUTPUT_DIR, "reconstructed.wav"), x_recon, SR)

# 波形統計をターミナル出力
print(f"入力波形: mean={np.mean(x_in):.4f}, std={np.std(x_in):.4f}, max={np.max(x_in):.4f}")
print(f"再構成波形: mean={np.mean(x_recon):.4f}, std={np.std(x_recon):.4f}, max={np.max(x_recon):.4f}")

# ざっくり波形をテキスト表示（先頭100サンプル）
start = SR  # 1秒後
end = start + 200

print(f"\n--- 入力波形サンプル（{start/SR:.1f}〜{end/SR:.1f}秒付近, 200点）---")
print(np.array2string(x_in[start:end], precision=3, separator=", "))

print(f"\n--- 再構成波形サンプル（{start/SR:.1f}〜{end/SR:.1f}秒付近, 200点）---")
print(np.array2string(x_recon[start:end], precision=3, separator=", "))

print("\n✅ 再構成波形を保存しました -> reconstructed/input.wav, reconstructed/reconstructed.wav")