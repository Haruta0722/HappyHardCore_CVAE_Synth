import tensorflow as tf
import numpy as np
import librosa

# 学習時と同じものを import（超重要）
from train import (
    build_encoder,
    build_decoder,
    WaveTimeConditionalCVAE,
    LATENT_DIM,
    ENC_CHANNELS,
    DEC_CHANNELS,
    DOWNSAMPLE_FACTORS,
    write_wav
)

def infer(input_wav_name, out_name, cond):

    # ====== 音声読み込み ======
    wav, sr = librosa.load(input_wav_name, sr=32000)
    wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
    T = wav.shape[1]
    print(f"読み込み完了: {input_wav_name} (len={T})")

    # ====== モデル構築（学習時のコードと完全に同じ） ======
    encoder = build_encoder(
        latent_dim=LATENT_DIM,
        channels=ENC_CHANNELS,
        down_factors=DOWNSAMPLE_FACTORS
    )

    decoder = build_decoder(
        latent_dim=LATENT_DIM,
        channels=DEC_CHANNELS,
        up_factors=DOWNSAMPLE_FACTORS[::-1],
        cond_dim=4
    )

    model = WaveTimeConditionalCVAE(
        encoder, decoder,
        latent_dim=LATENT_DIM,
        kl_anneal_steps=1,     # 推論時は関係ない
        kl_weight_max=1.0,     # 推論時は関係ない
        free_bits=0.0          # 推論時は関係ない
    )

    # ====== モデル build（学習時と同じ呼び方で）=====
    x_in = tf.constant(wav, dtype=tf.float32)
    y_dummy = tf.zeros_like(x_in)
    lx = tf.constant([T], dtype=tf.int32)
    ly = tf.constant([T], dtype=tf.int32)

    _ = model([x_in, cond, y_dummy])  # build only

    # ====== 重みロード ======
    ckpt = "checkpoints_cvae/cvae_044.weights.h5"
    print(f"重みをロード: {ckpt}")
    model.load_weights(ckpt)
    print("重みロード完了")

    # ====== 推論（Encoder → sampling → Decoder） ======
    mu, logvar = model.encoder(x_in, training=False)
    std = tf.exp(0.5 * logvar)
    eps = tf.random.normal(shape=std.shape)
    z = mu + eps * std  # 再パラメータ化トリっク
    y_pred = model.decoder([z, cond], training=False)

    y_pred = y_pred.numpy()[0, :, 0]

    # ====== 保存 ======
    write_wav(out_name, y_pred, sr=32000)
    print(f"生成完了 → {out_name}")

    return y_pred


if __name__ == "__main__":
    cond = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    infer("datasets/input_data/0013.wav", "y1.wav", cond)
