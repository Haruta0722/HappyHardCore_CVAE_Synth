import librosa
import numpy as np

wave_path = "output.wav"
y, sr = librosa.load(wave_path, sr=None)

print(f"波形 shape: {y.shape}, サンプリングレート: {sr}")
print("先頭1000サンプル:")
print(y[:1000])
print(f"min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}")