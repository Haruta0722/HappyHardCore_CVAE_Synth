import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf


def analyze_harmonic_structure(wave, sr=48000, expected_freq=440.0):
    """
    波形の倍音構造を分析
    """
    # FFT
    fft = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1 / sr)
    magnitude = np.abs(fft)

    # デシベルに変換
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # 期待される倍音の位置
    harmonics = []
    for h in range(1, 17):  # 16倍音まで
        harmonic_freq = expected_freq * h
        # 最も近い周波数ビンを探す
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        actual_freq = freqs[idx]
        power = magnitude_db[idx]
        harmonics.append((h, harmonic_freq, actual_freq, power))

    return freqs, magnitude_db, harmonics


def check_dataset_harmonics(dataset_path="dataset.csv"):
    """
    データセットの音が本当に倍音構造を持っているか確認
    """
    print("=" * 60)
    print("データセット倍音構造チェック")
    print("=" * 60)

    # サンプルを読み込み
    import pandas as pd

    df = pd.read_csv(dataset_path)

    print(f"\nデータセット: {len(df)} サンプル")
    print(f"列: {df.columns.tolist()}")

    # 最初の数サンプルを分析
    n_samples = min(5, len(df))

    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4 * n_samples))

    for i in range(n_samples):
        row = df.iloc[i]

        # ファイルパスまたは波形データを取得
        # （データセットの構造に応じて調整が必要）
        if "file_path" in row:
            wave, sr = sf.read(row["file_path"])
        elif "wave" in row:
            wave = row["wave"]
            sr = 48000
        else:
            print(f"⚠️  サンプル {i}: 波形データが見つかりません")
            continue

        # 音高情報
        if "pitch" in row:
            pitch_midi = row["pitch"]
            expected_freq = 440.0 * 2 ** ((pitch_midi - 69) / 12)
        else:
            expected_freq = 440.0
            pitch_midi = 69

        # 音色情報
        if "screech" in row:
            timbre = f"s={row['screech']:.2f}, a={row['acid']:.2f}, p={row['pluck']:.2f}"
        else:
            timbre = "unknown"

        # 倍音分析
        freqs, magnitude_db, harmonics = analyze_harmonic_structure(
            wave, sr, expected_freq
        )

        # 波形プロット
        axes[i, 0].plot(wave[:4800])
        axes[i, 0].set_title(f"Sample {i}: MIDI={pitch_midi:.0f}, {timbre}")
        axes[i, 0].set_xlabel("Sample")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)

        # スペクトルプロット
        axes[i, 1].plot(freqs[:2000], magnitude_db[:2000], alpha=0.7)

        # 倍音の位置をマーク
        for h, expected_f, actual_f, power in harmonics[:8]:
            if expected_f < 2000:
                axes[i, 1].axvline(
                    expected_f, color="r", linestyle="--", alpha=0.3
                )
                axes[i, 1].text(expected_f, power, f"H{h}", fontsize=8)

        axes[i, 1].set_title(f"Spectrum (F0={expected_freq:.1f}Hz)")
        axes[i, 1].set_xlabel("Frequency [Hz]")
        axes[i, 1].set_ylabel("Magnitude [dB]")
        axes[i, 1].grid(True)
        axes[i, 1].set_xlim([0, 2000])

        # 倍音の存在を判定
        print(f"\nサンプル {i} (MIDI={pitch_midi:.0f}):")
        print("  倍音構造:")

        fundamental_power = harmonics[0][3]  # 基本波のパワー
        has_harmonics = False

        for h, expected_f, actual_f, power in harmonics[:8]:
            relative_power = power - fundamental_power
            if h == 1:
                status = "✓ 基本波"
            elif relative_power > -20:  # 基本波の-20dB以内
                status = "✓ 存在"
                has_harmonics = True
            else:
                status = "✗ 弱い/なし"

            print(f"    {h}倍音 ({expected_f:.0f}Hz): {power:.1f}dB ({status})")

        if not has_harmonics:
            print("  ⚠️  WARNING: 倍音がほとんどありません（正弦波に近い）")

    plt.tight_layout()
    plt.savefig("dataset_harmonic_analysis.png", dpi=150)
    print(f"\n✓ 分析結果を保存: dataset_harmonic_analysis.png")

    print("\n" + "=" * 60)
    print("診断結果")
    print("=" * 60)

    print(
        """
もし倍音がほとんど検出されない場合:
  → データセットが単純な正弦波で構成されている
  → モデルは倍音構造を学習できない
  → データ生成を見直す必要がある

推奨されるデータ構造:
  - 基本波 + 2-8倍音
  - 倍音の振幅は音色によって変化
  - screech: 高次倍音が強い
  - acid: 中域倍音が強い
  - pluck: 低次倍音のみ
    """
    )

    print("=" * 60)


def create_harmonic_rich_dataset(n_samples=100):
    """
    倍音豊かなテストデータセットを生成
    """
    print("\n" + "=" * 60)
    print("倍音豊かなテストデータセット生成")
    print("=" * 60)

    import pandas as pd

    data = []
    TIME_LENGTH = int(1.3 * 48000)

    for i in range(n_samples):
        # ランダムな音高
        midi = np.random.randint(36, 72)
        pitch_norm = (midi - 36.0) / 35.0
        freq = 440.0 * 2 ** ((midi - 69) / 12.0)

        t = np.arange(TIME_LENGTH) / 48000.0

        # ランダムに音色を選択
        timbre_type = np.random.choice(["screech", "acid", "pluck"])

        if timbre_type == "screech":
            # 高周波倍音が強い、鋭い音
            wave = (
                1.0 * np.sin(2 * np.pi * freq * t)  # 基本波
                + 0.6 * np.sin(2 * np.pi * freq * 2 * t)  # 2倍音
                + 0.5 * np.sin(2 * np.pi * freq * 3 * t)  # 3倍音
                + 0.4 * np.sin(2 * np.pi * freq * 4 * t)  # 4倍音
                + 0.3 * np.sin(2 * np.pi * freq * 5 * t)  # 5倍音
                + 0.25 * np.sin(2 * np.pi * freq * 6 * t)  # 6倍音
                + 0.2 * np.sin(2 * np.pi * freq * 7 * t)  # 7倍音
                + 0.15 * np.sin(2 * np.pi * freq * 8 * t)
            )  # 8倍音
            envelope = np.exp(-t * 2)
            cond = (1, 0, 0)

        elif timbre_type == "acid":
            # 中域倍音が強い
            wave = (
                1.0 * np.sin(2 * np.pi * freq * t)
                + 0.7 * np.sin(2 * np.pi * freq * 2 * t)
                + 0.6 * np.sin(2 * np.pi * freq * 3 * t)
                + 0.4 * np.sin(2 * np.pi * freq * 4 * t)
                + 0.2 * np.sin(2 * np.pi * freq * 5 * t)
            )
            envelope = np.exp(-t * 1.5)
            cond = (0, 1, 0)

        else:  # pluck
            # 低次倍音のみ、柔らかい
            wave = (
                1.0 * np.sin(2 * np.pi * freq * t)
                + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
                + 0.25 * np.sin(2 * np.pi * freq * 3 * t)
            )
            envelope = np.exp(-t * 5)
            cond = (0, 0, 1)

        wave = wave * envelope
        wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.9

        # 保存
        filename = f"harmonic_test_{i:03d}_{timbre_type}.wav"
        sf.write(filename, wave, 48000)

        data.append(
            {
                "file_path": filename,
                "pitch": midi,
                "pitch_norm": pitch_norm,
                "screech": cond[0],
                "acid": cond[1],
                "pluck": cond[2],
                "timbre": timbre_type,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv("harmonic_test_dataset.csv", index=False)

    print(f"✓ {n_samples}サンプルの倍音豊かなデータセットを生成")
    print(f"  CSV: harmonic_test_dataset.csv")
    print(f"  WAV: harmonic_test_*.wav")

    # 生成したデータを分析
    print("\n生成したデータの倍音構造を確認:")
    check_dataset_harmonics("harmonic_test_dataset.csv")


if __name__ == "__main__":

    print("=" * 60)
    print("データセット倍音構造確認")
    print("=" * 60)

    print("\n1. 既存のデータセットを分析")
    print("2. 倍音豊かなテストデータセットを生成")
    print("\n選択: ")

    choice = "2"  # デフォルトで生成

    if choice == "1":
        check_dataset_harmonics("dataset.csv")
    else:
        create_harmonic_rich_dataset(n_samples=50)

    print("\n✓ 完了")
