
import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
import json

device = "cpu"

import torch
from BigVGAN.bigvgan import BigVGAN
from huggingface_hub import hf_hub_download
from BigVGAN.env import AttrDict
from RNN_model.model import build_encoder, build_decoder, TimeWiseCVAE 
from RNN_model.parameters import (
    LATENT_DIM,
    COND_DIM,
    MAX_FRAMES,
    N_FFT,
    HOP,
    N_MELS,
    SR,
    EX_FRAMES
)
# instantiate the model. You can opti

def wav_to_mel_py(wav_np):
    # wav_np: np.ndarray, dtype float32, shape (T,) or (T,1)
    if wav_np is None:
        return np.zeros((0, N_MELS), dtype=np.float32)

    wav_np = np.asarray(wav_np, dtype=np.float32).squeeze()
    if wav_np.ndim == 0:
        # スカラーになってしまっているなら空スペクトログラムを返す（保険）
        return np.zeros((0, N_MELS), dtype=np.float32)

    # librosa.stft: shape (freq_bins, frames)
    S = librosa.stft(
        wav_np, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, center=True
    )
    mag = np.abs(S).T  # -> (frames, freq_bins)

    # MEL_MAT_np must be shape (freq_bins, N_MELS)
    mel_basis = librosa.filters.mel(
        sr=SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=SR/2,
    )  # (n_mels, freq_bins)

    mel = np.dot(mag, mel_basis.T)  # (frames, n_mels)
    # log scaling
    mel = np.log(mel + 1e-5).astype(np.float32)
    mel = np.clip(mel, -11.5, 2.0)
    mel[np.isnan(mel)] = 0.0
    mel[np.isinf(mel)] = 0.0
    print(mel.min(), mel.max(), mel.mean(), mel.std())
    return mel


def infer(input_wav_name, out_name, cond):

    # ====== 音声読み込み ======
    wav, sr = librosa.load(input_wav_name, sr=SR)
    wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
    T = wav.shape[1]
    print(f"読み込み完了: {input_wav_name} (len={T})")

    # ====== モデル構築（学習時のコードと完全に同じ） ======
    encoder = build_encoder()
    decoder = build_decoder()
    cvae = TimeWiseCVAE(encoder, decoder)

    # ====== モデル build（学習時と同じ呼び方で）=====
    frame_len = MAX_FRAMES # 1024
    x_in = wav_to_mel_py(wav[0, :, 0])  # (T, n_mels)
    T = x_in.shape[0]

    if T < frame_len:
        pad_len = frame_len - T
        x_in = np.pad(
            x_in,
            ((0, pad_len), (0, 0)),
            mode="constant"
        )
    else:
        x_in = x_in[:frame_len]

    x_in = x_in[None, :, :]  # (1, 1024, 128)

    x_tf = tf.convert_to_tensor(x_in, dtype=tf.float32)
    mask_np = np.zeros((1, frame_len), dtype=np.float32)
    mask_np[0, :T + EX_FRAMES] = 1.0   # NumPyならOK
    mask_tf = tf.convert_to_tensor(mask_np, dtype=tf.float32)
    cond_tf = tf.convert_to_tensor(cond, dtype=tf.float32)  

    _ = cvae([x_tf, mask_tf, cond_tf])  # build only

    # ====== 重みロード ======
    ckpt = "RNN_model/checkpoints/mel_cvae_epoch_015.weights.h5"
    print(f"重みをロード: {ckpt}")
    cvae.load_weights(ckpt)
    print("重みロード完了")

    config_path = hf_hub_download(
    repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
    filename="config.json"
    )
    ckpt_path = hf_hub_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        filename="bigvgan_generator.pt"
    )

    with open(config_path, "r") as f:
        h = AttrDict(json.load(f))

    # instantiate
    model_gan = BigVGAN(h, use_cuda_kernel=False)
    gan_ckpt = torch.load(ckpt_path, map_location="cpu")
    print(list(gan_ckpt["generator"].keys())[:20])
    state_dict = gan_ckpt["generator"]
    model_gan.load_state_dict(state_dict)
    model_gan.remove_weight_norm()
    model_gan.eval().to(device)



    # compute mel spectrogram from the ground truth audio



    # ====== 推論 ======
    print("推論中...")
    z_mean, z_logvar = cvae.encoder([x_tf,mask_tf, cond_tf])
    z_std = tf.exp(0.5 * z_logvar)
    eps = tf.random.normal(shape=z_std.shape)
    z = z_mean + eps * z_std  # 再パラメータ化

    y_pred = cvae.decoder([z, cond_tf])
    mel = y_pred.numpy()              # (1,T, n_mels)
    mel = np.transpose(mel, (0, 2, 1))  # ✅ (1, n_mels, T)
    mel = torch.from_numpy(mel)       # torch.Tensor
    mel = mel.to(device=device, dtype=torch.float32)
        # generate waveform from mel
    with torch.inference_mode():
        wav_gen = model_gan(mel) # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]  
    wav_gen = wav_gen.squeeze()     # ✅ (T,)
    wav_out = wav_gen.cpu().numpy()

    sf.write(
        out_name,
        wav_out,
        SR,
        subtype="PCM_16"  # ここで16bit化
)


if __name__ == "__main__":
    cond = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    infer("datasets/input_data/0013.wav", "y1.wav", cond)