import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
import soundfile as sf
from create_datasets import make_dataset_from_synth_csv


def check_dataset_quality(dataset):
    """
    学習データの品質をチェック
    screech, acid, pluck の音色が本当に異なっているか確認
    """
    print("=" * 60)
    print("学習データ品質チェック")
    print("=" * 60)

    # 各ラベルごとにサンプルを収集
    screech_samples = []
    acid_samples = []
    pluck_samples = []
    dataset = dataset.unbatch()

    for wave, cond in dataset.take(100):  # 最初の100サンプル
        pitch, s, a, p = cond.numpy()

        # 主要なラベルを判定
        if s > 0.5:
            screech_samples.append(wave)
        elif a > 0.5:
            acid_samples.append(wave)
        elif p > 0.5:
            pluck_samples.append(wave)

    print(f"サンプル数:")
    print(f"  screech: {len(screech_samples)}")
    print(f"  acid: {len(acid_samples)}")
    print(f"  pluck: {len(pluck_samples)}")

    if (
        len(screech_samples) == 0
        or len(acid_samples) == 0
        or len(pluck_samples) == 0
    ):
        print("\n⚠️  WARNING: 一部のラベルのサンプルが不足しています")
        return

    # 各ラベルの代表サンプル
    screech_rep = screech_samples[0].numpy().squeeze()
    acid_rep = acid_samples[0].numpy().squeeze()
    pluck_rep = pluck_samples[0].numpy().squeeze()

    # スペクトログラム計算
    def compute_spectrum(wave):
        f, t, Sxx = signal.spectrogram(wave, 48000, nperseg=2048)
        return f, t, Sxx

    f, _, Sxx_screech = compute_spectrum(screech_rep)
    _, _, Sxx_acid = compute_spectrum(acid_rep)
    _, _, Sxx_pluck = compute_spectrum(pluck_rep)

    # 平均スペクトルを計算
    avg_spectrum_screech = np.mean(Sxx_screech, axis=1)
    avg_spectrum_acid = np.mean(Sxx_acid, axis=1)
    avg_spectrum_pluck = np.mean(Sxx_pluck, axis=1)

    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 波形
    for i, (wave, label) in enumerate(
        [(screech_rep, "screech"), (acid_rep, "acid"), (pluck_rep, "pluck")]
    ):
        axes[0, i].plot(wave[:4800])
        axes[0, i].set_title(f"{label} - Waveform")
        axes[0, i].set_xlabel("Sample")
        axes[0, i].set_ylabel("Amplitude")

    # 平均スペクトル
    axes[1, 0].plot(
        f[:500],
        10 * np.log10(avg_spectrum_screech[:500] + 1e-10),
        label="screech",
    )
    axes[1, 0].plot(
        f[:500], 10 * np.log10(avg_spectrum_acid[:500] + 1e-10), label="acid"
    )
    axes[1, 0].plot(
        f[:500], 10 * np.log10(avg_spectrum_pluck[:500] + 1e-10), label="pluck"
    )
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Power [dB]")
    axes[1, 0].set_title("Average Spectrum Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # スペクトログラム比較
    axes[1, 1].pcolormesh(
        f[:500],
        np.arange(Sxx_screech.shape[1]),
        10 * np.log10(Sxx_screech[:500].T + 1e-10),
        shading="auto",
        cmap="viridis",
    )
    axes[1, 1].set_title("screech - Spectrogram")
    axes[1, 1].set_ylabel("Time Frame")

    axes[1, 2].pcolormesh(
        f[:500],
        np.arange(Sxx_pluck.shape[1]),
        10 * np.log10(Sxx_pluck[:500].T + 1e-10),
        shading="auto",
        cmap="viridis",
    )
    axes[1, 2].set_title("pluck - Spectrogram")

    plt.tight_layout()
    plt.savefig("dataset_quality_check.png", dpi=150)
    print("\n✓ 可視化を保存: dataset_quality_check.png")

    # 代表サンプルを保存
    sf.write("sample_screech.wav", screech_rep, 48000)
    sf.write("sample_acid.wav", acid_rep, 48000)
    sf.write("sample_pluck.wav", pluck_rep, 48000)
    print("✓ 代表サンプルを保存:")
    print("  - sample_screech.wav")
    print("  - sample_acid.wav")
    print("  - sample_pluck.wav")

    # 統計分析
    print("\n" + "=" * 60)
    print("スペクトル差分析")
    print("=" * 60)

    # スペクトルの相関係数
    corr_sa = np.corrcoef(avg_spectrum_screech[:500], avg_spectrum_acid[:500])[
        0, 1
    ]
    corr_sp = np.corrcoef(avg_spectrum_screech[:500], avg_spectrum_pluck[:500])[
        0, 1
    ]
    corr_ap = np.corrcoef(avg_spectrum_acid[:500], avg_spectrum_pluck[:500])[
        0, 1
    ]

    print(f"スペクトル相関係数:")
    print(f"  screech vs acid:  {corr_sa:.4f}")
    print(f"  screech vs pluck: {corr_sp:.4f}")
    print(f"  acid vs pluck:    {corr_ap:.4f}")

    # 高周波エネルギー比較
    hf_energy_screech = np.sum(avg_spectrum_screech[100:500])
    hf_energy_acid = np.sum(avg_spectrum_acid[100:500])
    hf_energy_pluck = np.sum(avg_spectrum_pluck[100:500])

    print(f"\n高周波エネルギー (100-500 bins):")
    print(f"  screech: {hf_energy_screech:.2e}")
    print(f"  acid:    {hf_energy_acid:.2e}")
    print(f"  pluck:   {hf_energy_pluck:.2e}")

    # 判定
    print("\n" + "=" * 60)
    print("診断結果")
    print("=" * 60)

    avg_corr = (corr_sa + corr_sp + corr_ap) / 3

    if avg_corr > 0.95:
        print("⚠️  CRITICAL: 音色の違いが極めて小さい")
        print("   すべてのラベルでほぼ同じスペクトルです")
        print("   → モデルは音色の違いを学習できません")
        print("\n推奨対策:")
        print("   1. データ生成を見直す（本当に異なる音色か？）")
        print("   2. ラベルが正しく付与されているか確認")
        print("   3. より極端な音色の違いを作る")
    elif avg_corr > 0.85:
        print("⚠️  WARNING: 音色の違いが小さい")
        print("   わずかな違いしかありません")
        print("   → モデルは学習できますが、効果は限定的")
        print("\n推奨対策:")
        print("   1. データ拡張で音色の違いを強調")
        print("   2. 条件付けを強化したモデルを使用")
    else:
        print("✓ 音色の違いは十分にあります")
        print("   モデルアーキテクチャの改善が効果的です")

    print("=" * 60)


def create_synthetic_dataset_with_timbre():
    """
    音色の違いが明確な合成データセットを作成
    テスト用
    """
    print("\n" + "=" * 60)
    print("音色の違いが明確なテストデータセット生成")
    print("=" * 60)

    dataset = []
    TIME_LENGTH = int(1.3 * 48000)

    for i in range(30):
        # ランダムな音高
        midi = np.random.randint(36, 72)
        pitch_norm = (midi - 36.0) / 35.0
        freq = 440.0 * 2 ** ((midi - 69) / 12.0)

        t = np.arange(TIME_LENGTH) / 48000.0

        # ランダムに音色を選択
        timbre_type = np.random.choice(["screech", "acid", "pluck"])

        if timbre_type == "screech":
            # 高周波が強い、鋭い音
            wave = (
                0.5 * np.sin(2 * np.pi * freq * t)
                + 0.4 * np.sin(2 * np.pi * freq * 3 * t)
                + 0.3 * np.sin(2 * np.pi * freq * 5 * t)
                + 0.2 * np.sin(2 * np.pi * freq * 7 * t)
            )
            # 鋭いアタック
            envelope = np.exp(-t * 2)
            cond = (1, 0, 0)

        elif timbre_type == "acid":
            # 中域が強い、うねる音
            wave = 0.5 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(
                2 * np.pi * freq * 2 * t + np.sin(2 * np.pi * 5 * t)
            )
            # LFOで変調
            lfo = 1 + 0.3 * np.sin(2 * np.pi * 3 * t)
            wave = wave * lfo
            envelope = np.exp(-t * 1)
            cond = (0, 1, 0)

        else:  # pluck
            # 低周波中心、柔らかい音
            wave = 0.6 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(
                2 * np.pi * freq * 2 * t
            )
            # 速い減衰
            envelope = np.exp(-t * 5)
            cond = (0, 0, 1)

        wave = wave * envelope
        wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.9

        dataset.append(
            (
                wave[:, None].astype(np.float32),
                np.array([pitch_norm, *cond], dtype=np.float32),
            )
        )

    print(f"✓ {len(dataset)}サンプルの合成データセットを生成")

    return dataset


if __name__ == "__main__":
    # オプション1: 既存のデータセットをチェック
    print("既存のデータセットをロードして分析しますか？ (y/n)")
    # response = input().lower()

    # if response == 'y':
    #     from create_datasets import load_dataset
    #     dataset = load_dataset()
    #     check_dataset_quality(dataset)
    # else:

    # オプション2: テスト用の合成データセットを作成して分析
    print("\nテスト用の合成データセットを生成して分析します...")
    test_dataset = make_dataset_from_synth_csv("dataset.csv")
    check_dataset_quality(test_dataset)

    print("\n" + "=" * 60)
    print("分析完了")
    print("生成されたファイルを確認してください")
    print("=" * 60)
