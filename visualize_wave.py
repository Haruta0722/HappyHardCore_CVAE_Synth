import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

gen_path = "y1.wav"  # 生成音
ref_path = "datasets/input_data/0013.wav"  # 教師データ

# 読み込み
y_gen, sr_gen = librosa.load(gen_path, sr=None, mono=True)
y_ref, sr_ref = librosa.load(ref_path, sr=None, mono=True)

# STFT 計算
D_gen = librosa.amplitude_to_db(
    np.abs(librosa.stft(y_gen, n_fft=1024, hop_length=256)), ref=np.max
)
D_ref = librosa.amplitude_to_db(
    np.abs(librosa.stft(y_ref, n_fft=1024, hop_length=256)), ref=np.max
)

plt.figure(figsize=(14, 8))

# 波形
plt.subplot(2, 2, 1)
librosa.display.waveshow(y_ref, sr=sr_ref)
plt.title("教師データ 波形")

plt.subplot(2, 2, 2)
librosa.display.waveshow(y_gen, sr=sr_gen)
plt.title("生成音 波形")

# スペクトログラム
plt.subplot(2, 2, 3)
librosa.display.specshow(
    D_ref, sr=sr_ref, hop_length=256, x_axis="time", y_axis="log", cmap="magma"
)
plt.colorbar(format="%+2.0f dB")
plt.title("教師データ スペクトログラム")

plt.subplot(2, 2, 4)
librosa.display.specshow(
    D_gen, sr=sr_gen, hop_length=256, x_axis="time", y_axis="log", cmap="magma"
)
plt.colorbar(format="%+2.0f dB")
plt.title("生成音 スペクトログラム")

plt.tight_layout()
plt.show()
