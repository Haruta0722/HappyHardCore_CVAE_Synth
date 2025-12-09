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

device = 'cuda'

import torch
import BigVGAN.bigvgan as bigvgan
import librosa
from meldataset import get_mel_spectrogram

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

# load wav file and compute mel spectrogram
wav_path = '/path/to/your/audio.wav'
wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]

# compute mel spectrogram from the ground truth audio
mel = get_mel_spectrogram(wav, model.h).to(device) # mel is FloatTensor with shape [B(1), C_mel, T_frame]

# generate waveform from mel
with torch.inference_mode():
    wav_gen = model(mel) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
wav_gen_float = wav_gen.squeeze(0).cpu() # wav_gen is FloatTensor with shape [1, T_time]

# you can convert the generated waveform to 16 bit linear PCM
wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype('int16') # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype


if __name__ == "__main__":
    cond = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    infer("datasets/input_data/0013.wav", "y1.wav", cond)
