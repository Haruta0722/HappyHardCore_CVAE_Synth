# infer_griffin.py
from train_spectol import (
    build_encoder,
    build_decoder,
    MelTimeCVAE,
    LATENT_DIM,
    COND_DIM,
    N_FFT,
    HOP,
    N_MELS,
    SR,
)
import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
import json



device = "cpu"

# -------------------------
# Griffin-Lim helper
# -------------------------
def mel_to_wave_via_griffinlim(
    mel_spec_linear,  # expected: numpy array shape (n_mels, T) — linear magnitude (not log)
    sr,
    n_fft,
    hop_length,
    win_length,
    n_iter=60,
    n_mels=None,
):
    """
    mel_spec_linear: (n_mels, T) - linear mel magnitudes (NOT dB, NOT log)
    Returns: waveform y (1D numpy float32)
    """
    # safety cast
    mel_spec = np.asarray(mel_spec_linear, dtype=np.float32)
    if mel_spec.ndim != 2:
        raise ValueError("mel_spec_linear must be 2D (n_mels, T)")

    # librosa.feature.inverse.mel_to_stft expects shape (n_mels, t) -> returns linear-frequency STFT magnitude
    # We use power=1.0 because mel_spec is magnitude (not power)
    S = librosa.feature.inverse.mel_to_stft(
        mel_spec, sr=sr, n_fft=n_fft, power=1.0
    )  # shape: (freq_bins, frames)

    # S is magnitude (linear). librosa.griffinlim expects magnitude
    y = librosa.griffinlim(
        S,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )
    return y.astype(np.float32)


# -------------------------
# wav_to_mel_py (you provided)
# -------------------------
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
        fmax=SR / 2,
    )  # (n_mels, freq_bins)

    mel = np.dot(mag, mel_basis.T)  # (frames, n_mels)
    # log scaling -> this is what your training code used
    mel = np.log(mel + 1e-5).astype(np.float32)
    mel = np.clip(mel, -11.5, 2.0)
    mel[np.isnan(mel)] = 0.0
    mel[np.isinf(mel)] = 0.0
    print("wav_to_mel_py:", mel.min(), mel.max(), mel.mean(), mel.std())
    return mel


# -------------------------
# infer (Griffin-Lim path)
# -------------------------
def infer_griffin(input_wav_name, out_name, cond, n_iter=60):
    # ====== 音声読み込み ======
    wav, sr = librosa.load(input_wav_name, sr=SR)
    wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
    T = wav.shape[1]
    print(f"読み込み完了: {input_wav_name} (len={T})")



    # ====== モデル build（学習時と同じ呼び方で）=====
    x_in = wav_to_mel_py(wav[0, :, 0])  # (frames, n_mels)
    x_in = x_in[None, ...]  # (1, frames, n_mels)
    

    # ---- convert to (n_mels, T) linear magnitude ----
    # your wav_to_mel_py used np.log(mel + 1e-5) during encoding, so decoder likely outputs that same log-scale.
    # therefore inverse is np.exp(...) - 1e-5 (we'll clip to >= 1e-10)
    mel_pred = mel_pred[0]  # (frames, n_mels)
    # numerical safety:
    mel_pred = np.clip(mel_pred, -50.0, 50.0)
    mel_linear = np.exp(mel_pred)  # -> (frames, n_mels), linear magnitude
    mel_linear = np.maximum(mel_linear, 1e-10)

    # transpose to (n_mels, T) as required by mel_to_wave_via_griffinlim
    mel_linear_nmels_T = mel_linear.T  # (n_mels, T)

    # ---- Griffin-Lim reconstruction ----
    y = mel_to_wave_via_griffinlim(
        mel_linear_nmels_T,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        n_iter=n_iter,
        n_mels=N_MELS,
    )

    # ---- save ----
    sf.write(out_name, y, SR, subtype="PCM_16")
    print(f"書き出し完了: {out_name} (len={len(y)})")


# -------------------------
# If you want to keep original BigVGAN path as optional, you could implement infer_bigvgan similarly.
# But this script focuses on griffin-lim.
# -------------------------

if __name__ == "__main__":
    # dummy condition same as your example
    cond = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    # n_iter can be increased to 80~100 for slightly better phase, but slower.
    infer_griffin("datasets/input_data/0013.wav", "y1_griffin.wav", cond, n_iter=80)