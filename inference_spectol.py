from train_spectol import (
    build_encoder,
    build_decoder,
   MelTimeCVAE,
    LATENT_DIM,
    COND_DIM,
    N_FFT,
    HOP,
    N_MELS,
    SR 
)
import tensorflow as tf
import librosa
from griffin_lim import mel_to_wave_via_griffinlim
import numpy as np

import numpy as np
import librosa

# 前提：SR, N_FFT, HOP, MEL_PINV_np が既に定義済みであること

def wav_to_mel_py(wav_np):
    # wav_np: np.ndarray, dtype float32, shape (T,) or (T,1)
    if wav_np is None:
        return np.zeros((0, N_MELS), dtype=np.float32)

    wav_np = np.asarray(wav_np, dtype=np.float32).squeeze()
    if wav_np.ndim == 0:
        # スカラーになってしまっているなら空スペクトログラムを返す（保険）
        return np.zeros((0, N_MELS), dtype=np.float32)

    # librosa.stft: shape (freq_bins, frames)
    S = librosa.stft(wav_np, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, center=True)
    mag = np.abs(S).T  # -> (frames, freq_bins)


    # MEL_MAT_np must be shape (freq_bins, N_MELS)
    mel = np.matmul(mag, N_MELS)  # (frames, n_mels)
    # log scaling
    mel = np.log(mel + 1e-6).astype(np.float32)

    mel[np.isnan(mel)] = 0.0
    mel[np.isinf(mel)] = 0.0
    return mel

def mel_log_to_linear_mag(mel_log, mel_pinv_np=80):
    """
    mel_log: np.ndarray, shape (T, n_mels) or (B, T, n_mels)
    Returns linear magnitude spectrogram in shape (T, freq_bins) or (B, T, freq_bins)
    """
    # convert to numpy in case tf.Tensor passed
    if hasattr(mel_log, "numpy"):
        mel_log = mel_log.numpy()
    mel_log = np.asarray(mel_log, dtype=np.float32)

    # mel linear
    mel_lin = np.exp(mel_log)  # (T, n_mels) or (B, T, n_mels)
    # ensure no extremely small values
    mel_lin = np.maximum(mel_lin, 1e-7)

    # multiply with pseudo-inverse to get linear spectrogram
    # mel_pinv_np: shape (n_mels, freq_bins)
    if mel_lin.ndim == 2:
        # (T, n_mels) -> (T, freq_bins)
        S = mel_lin @ mel_pinv_np
        return S  # shape (T, freq_bins)
    elif mel_lin.ndim == 3:
        # (B, T, n_mels) -> (B, T, freq_bins)
        B = mel_lin.shape[0]
        T = mel_lin.shape[1]
        # Efficient matmul over last axis
        S = np.matmul(mel_lin, mel_pinv_np)  # (B, T, freq_bins)
        return S
    else:
        raise ValueError("mel_log must be shape (T,n_mels) or (B,T,n_mels)")

def infer(input_wav_name, out_name, cond):

    # ====== 音声読み込み ======
    wav, sr = librosa.load(input_wav_name, sr=SR)
    wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
    T = wav.shape[1]
    print(f"読み込み完了: {input_wav_name} (len={T})")

    # ====== モデル構築（学習時のコードと完全に同じ） ======
    enc = build_encoder(latent_dim=LATENT_DIM, chs=[128,128,128])
    dec = build_decoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM, channels=[128,128,64],
                        n_dilated_per_stage=3, use_film_in_residual=True)
    model = MelTimeCVAE(enc, dec)

    # ====== モデル build（学習時と同じ呼び方で）=====
    x_in = wav_to_mel_py(wav[0,:,0])  # (T, n_mels)
    y_dummy = tf.zeros_like(x_in)

    _ = model([x_in, cond])  # build only

    # ====== 重みロード ======
    ckpt = "checkpoints/mel_cvae_epoch_030.weights.h5"
    print(f"重みをロード: {ckpt}")
    model.load_weights(ckpt)
    print("重みロード完了")

    # ====== 推論 ======
    print("推論中...")
    z_mean, z_logvar = model.encoder(x_in, training=False)
    z_std = tf.exp(0.5 * z_logvar)
    eps = tf.random.normal(shape=z_std.shape)
    z = z_mean + eps * z_std  # 再パラメータ化
    y_pred = model.decoder([z, cond], training=False)

    print("推論完了")
