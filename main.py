import gradio as gr
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
from train import WaveCVAE

SR = 32000

# ハイパーパラメータは学習時と同じにする
LATENT_DIM = 64
COND_DIM = 4
KL_WEIGHT = 1e-3  # 学習時に使った値

# モデル構築
model = WaveCVAE(
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    kl_weight=KL_WEIGHT,
)

# -------------------------------
# モデルを「ビルド」してから重みをロード
# -------------------------------
# ダミー入力でshapeを確定させる
dummy_audio = tf.zeros((1, 16000, 1), dtype=tf.float32)  # 長さは任意
dummy_cond = tf.zeros((1, COND_DIM), dtype=tf.float32)
output = model([dummy_audio, dummy_cond], training=False)  # ← ここが重要
print("出力 shape:", output.shape)
print("出力の平均値:", tf.reduce_mean(output).numpy())

# 学習済み重みをロード
model.load_weights("checkpoints/wavecvae.weights.h5")
print("✅ モデル重みをロードしました")

# 重み統計を出力
total_params = 0
for w in model.weights:
    mean_val = tf.reduce_mean(w).numpy()
    total_params += np.prod(w.shape)
    print(f"{w.name}: mean={mean_val:.5f}, shape={w.shape}")

print(f"総パラメータ数: {total_params}")


# 生成関数（エンコード→デコード）
def generate(input_wav: gr.File, cond1, cond2, cond3, cond4):
    if input_wav is None:
        return None
    y, sr = librosa.load(input_wav.name, sr=SR)
    y = np.expand_dims(y, axis=(0, -1))  # [1, T, 1]

    cond = np.array([[cond1, cond2, cond3, cond4]], dtype=np.float32)

    # エンコード
    mu, logvar = model.encoder([y, cond], training=False)
    eps = tf.random.normal(tf.shape(mu))
    z = mu + tf.exp(0.5 * logvar) * eps

    # デコード
    target_time = y.shape[1]
    y_pred = model.decoder(z, cond, target_time=target_time, training=False)
    y_pred = tf.squeeze(y_pred).numpy()

    sf.write("preview.wav", y_pred, SR)
    return "preview.wav"


ui = gr.Interface(
    fn=generate,
    inputs=[
        gr.File(label="Input Wav"),
        gr.Slider(0, 1, value=0.5, label="Attack"),
        gr.Slider(0, 1, value=0.5, label="Distortion"),
        gr.Slider(0, 1, value=0.5, label="Thickness"),
        gr.Slider(0, 1, value=0.5, label="Center_tone"),
    ],
    outputs=gr.Audio(label="Generated / Converted Audio"),
    title="HappyHardcore WaveCVAE Synth",
    description="WAVファイルを潜在空間に変換し、条件ベクトルで音色変換します。",
)
ui.launch()
