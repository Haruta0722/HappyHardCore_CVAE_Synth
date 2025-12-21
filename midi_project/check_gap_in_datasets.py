import librosa
import numpy as np
import glob

SR = 48000
FRAME_SIZE = 1024
HOP = 512
RMS_THRESH = 1e-4  # ← 無音とみなす閾値（要調整）

def analyze_silence(wav_path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_SIZE,
        hop_length=HOP
    )[0]

    silent_frames = rms < RMS_THRESH
    silence_ratio = silent_frames.mean()

    return silence_ratio, rms.mean(), rms.min(), rms.max()

# データセット全体を調べる
ratios = []
problem_files = []

for wav in glob.glob("datasets/**/*.wav", recursive=True):
    silence_ratio, rms_mean, rms_min, rms_max = analyze_silence(wav)
    ratios.append(silence_ratio)
    if silence_ratio > 0.5:
        problem_files.append(wav)

print("平均無音率:", np.mean(ratios))
print("最大無音率:", np.max(ratios))
print("最小無音率:", np.min(ratios))
print("無音率が高いファイル一覧:")
for f in problem_files:
    print(f + " ")
