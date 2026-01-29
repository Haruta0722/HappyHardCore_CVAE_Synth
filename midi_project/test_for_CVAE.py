import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import warnings
from create_datasets import load_wav, crop_or_pad

warnings.filterwarnings("ignore")

# 教師データと生成音のパス

gt = [
    "datasets/C3/0012.wav",
    "datasets/C3/0013.wav",
    "datasets/C3/0014.wav",
    "datasets/C3/0015.wav",
    "datasets/C3/0016.wav",
    "datasets/C3/0017.wav",
    "datasets/C3/0018.wav",
    "datasets/C3/0019.wav",
    "datasets/C3/0020.wav",
    "datasets/C3/0021.wav",
    "datasets/C3/0022.wav",
    "datasets/C3/0023.wav",
]

gen = [
    "test\c3_acid_01.wav",
    "test\c3_acid_02.wav",
    "test\c3_acid_03.wav",
    "test\c3_acid_04.wav",
    "test\c3_acid_05.wav",
]


def load_audio(file_path, sr=22050):
    """音声ファイルを読み込む"""
    try:
        y, _ = librosa.load(file_path, sr=sr)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def spectral_centroid_curve(y, sr):
    """スペクトル重心の時間変化を計算"""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(centroid, sr=sr)
    return times, centroid


def extract_harmonic_energy(y, sr, n_harmonics=20):
    """f0正規化した倍音エネルギーベクトルを抽出"""
    # f0を推定
    f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr)
    f0_median = np.median(f0[f0 > 0]) if np.any(f0 > 0) else 100

    # スペクトル計算
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # 各倍音のエネルギーを計算
    harmonic_energies = []
    for h in range(1, n_harmonics + 1):
        target_freq = f0_median * h
        # 最も近い周波数ビンを探す
        idx = np.argmin(np.abs(freqs - target_freq))
        # 周辺のビンも含めてエネルギーを計算
        start_idx = max(0, idx - 2)
        end_idx = min(len(freqs), idx + 3)
        energy = np.mean(D[start_idx:end_idx, :])
        harmonic_energies.append(energy)

    return np.array(harmonic_energies)


def mel_spectrogram_and_kl(y, sr, n_mels=128):
    """メルスペクトログラムとKLダイバージェンスを計算"""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def hilbert_envelope(y):
    """Hilbert変換による振幅包絡を計算"""
    analytic_signal = signal.hilbert(y)
    envelope = np.abs(analytic_signal)
    return envelope


def spectral_flux(y, sr):
    """スペクトルフラックスの時間変化を計算"""
    S = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    times = librosa.times_like(flux, sr=sr, hop_length=512)
    return times, flux


def calculate_mel_difference(mel_gt, mel_gen):
    """メルスペクトログラムの差分を計算"""
    # サイズを揃える
    min_time = min(mel_gt.shape[1], mel_gen.shape[1])
    mel_gt_trimmed = mel_gt[:, :min_time]
    mel_gen_trimmed = mel_gen[:, :min_time]

    # log差分を計算
    diff = np.abs(mel_gt_trimmed - mel_gen_trimmed)
    return diff


# データ読み込み
print("Loading audio files...")
gt_audios = [load_wav(path) for path in gt]
gen_audios = [load_wav(path) for path in gen]

# Noneを除外
gt_audios = [a for a in gt_audios if a is not None]
gen_audios = [a for a in gen_audios if a is not None]

if len(gt_audios) == 0 or len(gen_audios) == 0:
    print("Error: No valid audio files loaded")
    exit()

sr = 48000

# 出力ディレクトリの作成
import os

output_dir = "paper_figures"
os.makedirs(output_dir, exist_ok=True)

# ========================================
# 1. スペクトル重心の比較
# ========================================
print("Computing spectral centroid...")
gt_centroids = []
for y in gt_audios:
    times, centroid = spectral_centroid_curve(y, sr)
    gt_centroids.append(centroid)

gen_centroids = []
for y in gen_audios:
    times, centroid = spectral_centroid_curve(y, sr)
    gen_centroids.append(centroid)

# 平均を計算（長さを揃える）
min_len_gt = min([len(c) for c in gt_centroids])
min_len_gen = min([len(c) for c in gen_centroids])
gt_centroid_avg = np.mean([c[:min_len_gt] for c in gt_centroids], axis=0)
gen_centroid_avg = np.mean([c[:min_len_gen] for c in gen_centroids], axis=0)

times_gt = librosa.times_like(gt_centroid_avg, sr=sr)
times_gen = librosa.times_like(gen_centroid_avg, sr=sr)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(
    times_gt,
    gt_centroid_avg,
    label="Ground Truth (average)",
    linewidth=2,
    alpha=0.8,
    color="#2E86AB",
)
ax1.plot(
    times_gen,
    gen_centroid_avg,
    label="CVAE Generated (average)",
    linewidth=2,
    alpha=0.8,
    color="#A23B72",
)
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Spectral Centroid (Hz)", fontsize=12)
ax1.set_title("Spectral Centroid Comparison", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{output_dir}/01_spectral_centroid.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: {output_dir}/01_spectral_centroid.png")

# ========================================
# 2. 倍音構造の比較
# ========================================
print("Computing harmonic structure...")
gt_harmonics = []
for y in gt_audios:
    harmonics = extract_harmonic_energy(y, sr)
    gt_harmonics.append(harmonics)

gen_harmonics = []
for y in gen_audios:
    harmonics = extract_harmonic_energy(y, sr)
    gen_harmonics.append(harmonics)

gt_harmonics_avg = np.mean(gt_harmonics, axis=0)
gen_harmonics_avg = np.mean(gen_harmonics, axis=0)

harmonic_indices = np.arange(1, len(gt_harmonics_avg) + 1)

# 上下に2つの表を作成
fig2, (ax2_top, ax2_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Ground Truth
ax2_top.bar(
    harmonic_indices,
    gt_harmonics_avg,
    color="#2E86AB",
    alpha=0.8,
    edgecolor="black",
)
ax2_top.set_ylabel("Energy", fontsize=12)
ax2_top.set_title(
    "Ground Truth - Harmonic Structure", fontsize=13, fontweight="bold"
)
ax2_top.grid(True, alpha=0.3, axis="y")

# CVAE Generated
ax2_bottom.bar(
    harmonic_indices,
    gen_harmonics_avg,
    color="#A23B72",
    alpha=0.8,
    edgecolor="black",
)
ax2_bottom.set_xlabel("Harmonic Index", fontsize=12)
ax2_bottom.set_ylabel("Energy", fontsize=12)
ax2_bottom.set_title(
    "CVAE Generated - Harmonic Structure", fontsize=13, fontweight="bold"
)
ax2_bottom.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/02_harmonic_structure.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: {output_dir}/02_harmonic_structure.png")

# ========================================
# 3. メルスペクトログラムとKLダイバージェンス
# ========================================
print("Computing mel spectrograms...")
gt_mels = [mel_spectrogram_and_kl(y, sr) for y in gt_audios]
gen_mels = [mel_spectrogram_and_kl(y, sr) for y in gen_audios]

# GTの平均メルスペクトログラム
min_time_gt = min([m.shape[1] for m in gt_mels])
gt_mel_avg = np.mean([m[:, :min_time_gt] for m in gt_mels], axis=0)

# 各生成音とGT平均の差分を計算
kl_divergences = []
for gen_mel in gen_mels:
    diff = calculate_mel_difference(gt_mel_avg, gen_mel)
    kl_divergences.append(np.mean(diff))

# 差分が最小と最大のものを選択
min_idx = np.argmin(kl_divergences)
max_idx = np.argmax(kl_divergences)

# --- 最小差分のケース ---
fig3, axes3 = plt.subplots(3, 1, figsize=(10, 9))

# GT
librosa.display.specshow(
    gt_mel_avg, sr=sr, x_axis="time", y_axis="mel", ax=axes3[0], cmap="viridis"
)
axes3[0].set_title(
    "Ground Truth (average) - Min Divergence Case",
    fontsize=13,
    fontweight="bold",
)
axes3[0].set_ylabel("Mel Frequency", fontsize=11)

# Generated (Min)
min_time = min(gt_mel_avg.shape[1], gen_mels[min_idx].shape[1])
librosa.display.specshow(
    gen_mels[min_idx][:, :min_time],
    sr=sr,
    x_axis="time",
    y_axis="mel",
    ax=axes3[1],
    cmap="viridis",
)
axes3[1].set_title(
    f"CVAE Generated {min_idx+1} (Min Divergence: {kl_divergences[min_idx]:.4f})",
    fontsize=13,
    fontweight="bold",
)
axes3[1].set_ylabel("Mel Frequency", fontsize=11)

# 差分ヒートマップ（最小）
diff_min = calculate_mel_difference(gt_mel_avg, gen_mels[min_idx])
im1 = axes3[2].imshow(
    diff_min, aspect="auto", origin="lower", cmap="hot", interpolation="nearest"
)
axes3[2].set_title(
    "Difference |log(GT) - log(Gen)| - Min", fontsize=13, fontweight="bold"
)
axes3[2].set_xlabel("Time Frame", fontsize=11)
axes3[2].set_ylabel("Mel Bin", fontsize=11)
plt.colorbar(im1, ax=axes3[2], label="Difference (dB)")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/03_mel_spectrogram_min.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: {output_dir}/03_mel_spectrogram_min.png")

# --- 最大差分のケース ---
fig4, axes4 = plt.subplots(3, 1, figsize=(10, 9))

# GT
librosa.display.specshow(
    gt_mel_avg, sr=sr, x_axis="time", y_axis="mel", ax=axes4[0], cmap="viridis"
)
axes4[0].set_title(
    "Ground Truth (average) - Max Divergence Case",
    fontsize=13,
    fontweight="bold",
)
axes4[0].set_ylabel("Mel Frequency", fontsize=11)

# Generated (Max)
min_time = min(gt_mel_avg.shape[1], gen_mels[max_idx].shape[1])
librosa.display.specshow(
    gen_mels[max_idx][:, :min_time],
    sr=sr,
    x_axis="time",
    y_axis="mel",
    ax=axes4[1],
    cmap="viridis",
)
axes4[1].set_title(
    f"CVAE Generated {max_idx+1} (Max Divergence: {kl_divergences[max_idx]:.4f})",
    fontsize=13,
    fontweight="bold",
)
axes4[1].set_ylabel("Mel Frequency", fontsize=11)

# 差分ヒートマップ（最大）
diff_max = calculate_mel_difference(gt_mel_avg, gen_mels[max_idx])
im2 = axes4[2].imshow(
    diff_max, aspect="auto", origin="lower", cmap="hot", interpolation="nearest"
)
axes4[2].set_title(
    "Difference |log(GT) - log(Gen)| - Max", fontsize=13, fontweight="bold"
)
axes4[2].set_xlabel("Time Frame", fontsize=11)
axes4[2].set_ylabel("Mel Bin", fontsize=11)
plt.colorbar(im2, ax=axes4[2], label="Difference (dB)")

plt.tight_layout()
plt.savefig(
    f"{output_dir}/04_mel_spectrogram_max.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: {output_dir}/04_mel_spectrogram_max.png")

# ========================================
# 4. 振幅包絡（Hilbert envelope）
# ========================================
print("Computing amplitude envelope...")
gt_envelopes = []
for y in gt_audios:
    env = hilbert_envelope(y)
    gt_envelopes.append(env)

gen_envelopes = []
for y in gen_audios:
    env = hilbert_envelope(y)
    gen_envelopes.append(env)

min_len_gt = min([len(e) for e in gt_envelopes])
min_len_gen = min([len(e) for e in gen_envelopes])
gt_env_avg = np.mean([e[:min_len_gt] for e in gt_envelopes], axis=0)
gen_env_avg = np.mean([e[:min_len_gen] for e in gen_envelopes], axis=0)

times_gt = np.arange(len(gt_env_avg)) / sr
times_gen = np.arange(len(gen_env_avg)) / sr

fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.plot(
    times_gt,
    gt_env_avg,
    label="Ground Truth (average)",
    linewidth=2,
    alpha=0.8,
    color="#2E86AB",
)
ax5.plot(
    times_gen,
    gen_env_avg,
    label="CVAE Generated (average)",
    linewidth=2,
    alpha=0.8,
    color="#A23B72",
)
ax5.set_xlabel("Time (s)", fontsize=12)
ax5.set_ylabel("Amplitude", fontsize=12)
ax5.set_title(
    "Amplitude Envelope Comparison (Hilbert)", fontsize=14, fontweight="bold"
)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{output_dir}/05_amplitude_envelope.png", dpi=300, bbox_inches="tight"
)
plt.close()
print(f"Saved: {output_dir}/05_amplitude_envelope.png")

# ========================================
# 5. スペクトルフラックス
# ========================================
print("Computing spectral flux...")
gt_fluxes = []
for y in gt_audios:
    times, flux = spectral_flux(y, sr)
    gt_fluxes.append(flux)

gen_fluxes = []
for y in gen_audios:
    times, flux = spectral_flux(y, sr)
    gen_fluxes.append(flux)

min_len_gt = min([len(f) for f in gt_fluxes])
min_len_gen = min([len(f) for f in gen_fluxes])
gt_flux_avg = np.mean([f[:min_len_gt] for f in gt_fluxes], axis=0)
gen_flux_avg = np.mean([f[:min_len_gen] for f in gen_fluxes], axis=0)

times_gt = librosa.times_like(gt_flux_avg, sr=sr, hop_length=512)
times_gen = librosa.times_like(gen_flux_avg, sr=sr, hop_length=512)

fig6, ax6 = plt.subplots(figsize=(8, 4))
ax6.plot(
    times_gt,
    gt_flux_avg,
    label="Ground Truth (average)",
    linewidth=2,
    alpha=0.8,
    color="#2E86AB",
)
ax6.plot(
    times_gen,
    gen_flux_avg,
    label="CVAE Generated (average)",
    linewidth=2,
    alpha=0.8,
    color="#A23B72",
)
ax6.set_xlabel("Time (s)", fontsize=12)
ax6.set_ylabel("Spectral Flux", fontsize=12)
ax6.set_title("Spectral Flux Comparison", fontsize=14, fontweight="bold")
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/06_spectral_flux.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir}/06_spectral_flux.png")

# ========================================
# まとめ
# ========================================
print("\n" + "=" * 60)
print("Evaluation complete! All figures saved to 'paper_figures/' directory")
print("=" * 60)
print("\nGenerated files:")
print(f"  1. {output_dir}/01_spectral_centroid.png")
print(f"  2. {output_dir}/02_harmonic_structure.png")
print(f"  3. {output_dir}/03_mel_spectrogram_min.png")
print(f"  4. {output_dir}/04_mel_spectrogram_max.png")
print(f"  5. {output_dir}/05_amplitude_envelope.png")
print(f"  6. {output_dir}/06_spectral_flux.png")
print("\nKL Divergence Statistics:")
print(f"  Min: {kl_divergences[min_idx]:.4f} (Generated sample {min_idx+1})")
print(f"  Max: {kl_divergences[max_idx]:.4f} (Generated sample {max_idx+1})")
print(f"  Mean: {np.mean(kl_divergences):.4f}")
print(f"  Std: {np.std(kl_divergences):.4f}")
print("=" * 60)
