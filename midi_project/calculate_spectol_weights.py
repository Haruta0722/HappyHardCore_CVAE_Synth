import numpy as np
import soundfile as sf
from scipy.signal import stft


def calculate_spectol_weights(path: str):
    """
    入力音声のスペクトル重心を計算し、表示する
    """
    # wav 読み込み
    y, sr = sf.read(path)

    # モノラル化（必要なら）
    if y.ndim == 2:
        y = y.mean(axis=1)

    # STFT
    f, t, Zxx = stft(y, fs=sr, nperseg=1024, noverlap=512)

    # 振幅スペクトル
    magnitude = np.abs(Zxx)

    # スペクトル重心（各フレーム）
    centroid = np.sum(f[:, None] * magnitude, axis=0) / np.sum(
        magnitude, axis=0
    )

    # 時間平均
    centroid_mean = np.mean(centroid)

    return centroid_mean
