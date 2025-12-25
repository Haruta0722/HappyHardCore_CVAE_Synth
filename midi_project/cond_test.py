import soundfile as sf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def test_condition_sensitivity(model, pitch=60):
    """
    条件ベクトルが出力に影響を与えているかテスト
    """
    print("=" * 60)
    print("条件ベクトル感度テスト")
    print("=" * 60)

    from model import (
        generate_frequency_features,
        TIME_LENGTH,
        LATENT_STEPS,
        LATENT_DIM,
    )

    pitch_norm = (pitch - 36.0) / 35.0

    # テストする条件
    conditions = {
        "screech_only": (1, 0, 0),
        "acid_only": (0, 1, 0),
        "pluck_only": (0, 0, 1),
        "screech_acid": (0.5, 0.5, 0),
        "all_equal": (0.33, 0.33, 0.34),
    }

    # 参照音声（シンプルな正弦波）
    t = np.arange(TIME_LENGTH) / 48000.0
    midi = pitch
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    reference_wave = reference_wave[:, None].astype(np.float32)

    outputs = {}
    spectrograms = []

    for name, cond in conditions.items():
        print(f"\n[{name}] 生成中...")

        # 条件ベクトル
        cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

        # 参照からzを抽出
        ref_wave = tf.constant(reference_wave[None, :, :], dtype=tf.float32)
        z_mean, z_logvar = model.encoder([ref_wave, cond_vector])
        z = z_mean  # 平均値を使用

        # 周波数特徴
        freq_feat = generate_frequency_features(cond_vector[:, 0], TIME_LENGTH)

        # 生成
        x_hat = model.decoder([z, cond_vector, freq_feat])
        x_hat = tf.squeeze(x_hat).numpy()

        # 正規化
        max_val = np.max(np.abs(x_hat))
        if max_val > 1e-6:
            x_hat = x_hat / max_val * 0.95

        outputs[name] = x_hat

        # 保存
        filename = f"test_cond_{name}.wav"
        sf.write(filename, x_hat, samplerate=48000)
        print(f"  ✓ 保存: {filename}")

        # スペクトログラム計算
        f, t_spec, Sxx = signal.spectrogram(x_hat, 48000, nperseg=2048)
        spectrograms.append((name, f, t_spec, Sxx))

    # スペクトログラムを可視化
    print("\n[可視化] スペクトログラムを比較中...")
    fig, axes = plt.subplots(len(conditions), 1, figsize=(12, 10))

    for i, (name, f, t_spec, Sxx) in enumerate(spectrograms):
        ax = axes[i]
        ax.pcolormesh(
            t_spec,
            f[:500],
            10 * np.log10(Sxx[:500] + 1e-10),
            shading="auto",
            cmap="viridis",
            vmin=-80,
            vmax=0,
        )
        ax.set_ylabel("Freq [Hz]")
        ax.set_title(f"{name}")
        ax.set_ylim([0, 2000])

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig("condition_comparison.png", dpi=150)
    print("  ✓ 保存: condition_comparison.png")

    # 差分分析
    print("\n" + "=" * 60)
    print("差分分析")
    print("=" * 60)

    ref_output = outputs["pluck_only"]

    for name, output in outputs.items():
        if name == "pluck_only":
            continue

        # L2距離
        diff = np.sqrt(np.mean((output - ref_output) ** 2))

        # 相関係数
        corr = np.corrcoef(output, ref_output)[0, 1]

        print(f"{name:20s}: L2={diff:.6f}, Corr={corr:.6f}")

    print("\n判定:")
    max_diff = max(
        [
            np.sqrt(np.mean((outputs[name] - ref_output) ** 2))
            for name in outputs
            if name != "pluck_only"
        ]
    )

    if max_diff < 0.01:
        print("⚠️  CRITICAL: 条件ベクトルがほとんど無視されています")
        print("   すべての条件で同じ音が出ています")
        print("   → freq_feat が支配的になっている可能性が高い")
    elif max_diff < 0.1:
        print("⚠️  WARNING: 条件の影響が弱い")
        print("   音色の違いが小さいです")
    else:
        print("✓ 条件ベクトルが適切に機能しています")
        print("   異なる条件で異なる音が生成されています")

    print("=" * 60)


def test_freq_feat_dominance(model, pitch=60):
    """
    freq_featの支配度をテスト
    freq_featをゼロにした場合と比較
    """
    print("\n" + "=" * 60)
    print("freq_feat 支配度テスト")
    print("=" * 60)

    from model import (
        generate_frequency_features,
        TIME_LENGTH,
        LATENT_STEPS,
        LATENT_DIM,
    )

    pitch_norm = (pitch - 36.0) / 35.0
    cond = (0, 0, 1)  # pluck
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # 参照波形
    t = np.arange(TIME_LENGTH) / 48000.0
    midi = pitch
    freq = 440.0 * 2 ** ((midi - 69) / 12.0)
    reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
    reference_wave = reference_wave[:, None].astype(np.float32)

    ref_wave = tf.constant(reference_wave[None, :, :], dtype=tf.float32)
    z_mean, z_logvar = model.encoder([ref_wave, cond_vector])
    z = z_mean

    # テスト1: 通常のfreq_feat
    freq_feat_normal = generate_frequency_features(
        cond_vector[:, 0], TIME_LENGTH
    )
    x_normal = model.decoder([z, cond_vector, freq_feat_normal])
    x_normal = tf.squeeze(x_normal).numpy()

    # テスト2: freq_featをゼロに
    freq_feat_zero = tf.zeros_like(freq_feat_normal)
    x_zero = model.decoder([z, cond_vector, freq_feat_zero])
    x_zero = tf.squeeze(x_zero).numpy()

    # テスト3: freq_featをランダムノイズに
    freq_feat_noise = tf.random.normal(tf.shape(freq_feat_normal), stddev=0.1)
    x_noise = model.decoder([z, cond_vector, freq_feat_noise])
    x_noise = tf.squeeze(x_noise).numpy()

    # 保存
    sf.write(
        "test_freq_normal.wav",
        x_normal / np.max(np.abs(x_normal)) * 0.95,
        48000,
    )
    sf.write(
        "test_freq_zero.wav",
        x_zero / (np.max(np.abs(x_zero)) + 1e-8) * 0.95,
        48000,
    )
    sf.write(
        "test_freq_noise.wav",
        x_noise / (np.max(np.abs(x_noise)) + 1e-8) * 0.95,
        48000,
    )

    print("✓ 3種類のテスト音声を生成しました:")
    print("  - test_freq_normal.wav: 通常のfreq_feat")
    print("  - test_freq_zero.wav: freq_feat = 0")
    print("  - test_freq_noise.wav: freq_feat = ランダムノイズ")

    # 分析
    diff_zero = np.sqrt(np.mean((x_normal - x_zero) ** 2))
    diff_noise = np.sqrt(np.mean((x_normal - x_noise) ** 2))

    print(f"\nL2距離:")
    print(f"  normal vs zero:  {diff_zero:.6f}")
    print(f"  normal vs noise: {diff_noise:.6f}")

    print("\n判定:")
    if diff_zero < 0.01:
        print("✓ freq_featの影響は小さい（zと条件が支配的）")
    elif diff_zero < 0.1:
        print("⚠️  freq_featがある程度影響している")
    else:
        print("⚠️  CRITICAL: freq_featが支配的！")
        print("   freq_featをゼロにすると音が大きく変わります")
        print("   → モデルがfreq_featに依存しすぎています")

    print("=" * 60)


if __name__ == "__main__":
    from model import TimeWiseCVAE
    from create_datasets import MAX_LEN

    # モデル読み込み
    model = TimeWiseCVAE()
    model.build(
        [
            (None, MAX_LEN, 1),
            (None, 4),
        ]
    )

    ckpt_path = "weights/epoch_100.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデル読み込み完了: {ckpt_path}\n")

    # テスト実行
    test_condition_sensitivity(model, pitch=60)
    test_freq_feat_dominance(model, pitch=60)

    print("\n" + "=" * 60)
    print("診断完了")
    print("生成された音声ファイルとグラフを確認してください")
    print("=" * 60)
