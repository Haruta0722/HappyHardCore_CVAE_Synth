import soundfile as sf
import tensorflow as tf
import numpy as np
from model import TimeWiseCVAE, LATENT_DIM, generate_frequency_features
from create_datasets import MAX_LEN


def diagnose_latent_space(model, n_samples=50):
    """
    学習済みモデルの潜在空間を診断
    zの実際の分布を確認する
    """
    print("\n" + "=" * 60)
    print("潜在空間診断")
    print("=" * 60)

    # ダミーデータで潜在変数の統計を取得
    dummy_waves = tf.random.normal((n_samples, MAX_LEN, 1))
    dummy_conds = tf.random.uniform((n_samples, 4))

    z_means = []
    z_logvars = []

    for i in range(n_samples):
        wave = dummy_waves[i : i + 1]
        cond = dummy_conds[i : i + 1]
        z_mean, z_logvar = model.encoder([wave, cond])
        z_means.append(z_mean.numpy())
        z_logvars.append(z_logvar.numpy())

    z_means = np.concatenate(z_means, axis=0)
    z_logvars = np.concatenate(z_logvars, axis=0)

    # 統計計算
    mean_of_means = np.mean(z_means)
    std_of_means = np.std(z_means)
    mean_of_logvars = np.mean(z_logvars)

    print(f"z_mean の平均: {mean_of_means:.6f}")
    print(f"z_mean の標準偏差: {std_of_means:.6f}")
    print(f"z_logvar の平均: {mean_of_logvars:.6f}")
    print(f"推定される z の標準偏差: {np.exp(0.5 * mean_of_logvars):.6f}")

    print("\n診断結果:")
    if std_of_means < 0.01:
        print("⚠️  CRITICAL: Posterior Collapse! zがほぼ使われていません")
        print("   → 推論時はz=0を使うか、学習をやり直してください")
        return 0.0, True  # std, collapsed
    elif std_of_means < 0.1:
        print("⚠️  WARNING: zの分散が小さい（部分的Collapse）")
        print(f"   → 推論時は temperature={std_of_means:.3f} を使用推奨")
        return std_of_means, False
    else:
        print("✓ zは適切に活用されています")
        print(f"   → 推論時は temperature={std_of_means:.3f} を使用推奨")
        return std_of_means, False

    print("=" * 60 + "\n")


def inference_with_zero_z(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    output_name="generated_zero_z.wav",
):
    """
    z=0 で生成（Posterior Collapse時の対処法）
    条件ベクトルだけで生成
    """
    print(f"\n[生成] z=0 で生成中...")

    cond_vector = tf.constant([[pitch, *cond]], dtype=tf.float32)

    # ★重要: zをゼロに設定
    z = tf.zeros(shape=(1, MAX_LEN // 16, LATENT_DIM), dtype=tf.float32)

    pitch_tensor = cond_vector[:, 0]
    freq_feat = generate_frequency_features(pitch_tensor, MAX_LEN)

    x_hat = model.decoder([z, cond_vector, freq_feat])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def inference_with_learned_distribution(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    temperature=1.0,
    output_name="generated_learned_z.wav",
):
    """
    学習済み分布からサンプリング
    エンコーダーで学習したzの分布を使う
    """
    print(f"\n[生成] temperature={temperature:.3f} でサンプリング中...")

    cond_vector = tf.constant([[pitch, *cond]], dtype=tf.float32)

    # ★改善1: ダミー入力を使ってzの平均を推定
    # 本来は学習データから計算すべきだが、簡易的にランダム入力で近似
    dummy_wave = tf.random.normal((1, MAX_LEN, 1), stddev=0.1)
    z_mean_ref, z_logvar_ref = model.encoder([dummy_wave, cond_vector])

    # 学習済み分布に基づいてサンプリング
    z = tf.random.normal(
        shape=tf.shape(z_mean_ref),
        mean=tf.reduce_mean(z_mean_ref),  # 平均を維持
        stddev=temperature,  # temperatureで制御
    )

    pitch_tensor = cond_vector[:, 0]
    freq_feat = generate_frequency_features(pitch_tensor, MAX_LEN)

    x_hat = model.decoder([z, cond_vector, freq_feat])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def inference_from_reference(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    reference_wave=None,
    output_name="generated_from_ref.wav",
):
    """
    参照音声から潜在変数を抽出して生成
    最も安定した生成方法
    """
    print(f"\n[生成] 参照音声から潜在変数を抽出中...")

    if reference_wave is None:
        # 参照がない場合は、シンプルな正弦波を生成
        t = np.arange(MAX_LEN) / 48000.0
        midi = pitch * 35.0 + 36.0
        freq = 440.0 * 2 ** ((midi - 69.0) / 12.0)
        reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
        reference_wave = reference_wave[:, None].astype(np.float32)

    reference_wave = tf.constant(reference_wave[None, :, :], dtype=tf.float32)
    cond_vector = tf.constant([[pitch, *cond]], dtype=tf.float32)

    # 参照からzを抽出
    z_mean, z_logvar = model.encoder([reference_wave, cond_vector])
    z = z_mean  # 平均値を使用（より安定）

    pitch_tensor = cond_vector[:, 0]
    freq_feat = generate_frequency_features(pitch_tensor, MAX_LEN)

    x_hat = model.decoder([z, cond_vector, freq_feat])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def main():
    print("=" * 60)
    print("改善版推論スクリプト")
    print("=" * 60)

    # モデル読み込み
    model = TimeWiseCVAE()
    model.build(
        [
            (None, MAX_LEN, 1),
            (None, 4),
        ]
    )

    ckpt_path = "checkpoints/epoch_100.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデルの重みを読み込みました: {ckpt_path}")

    # 潜在空間を診断
    temperature, is_collapsed = diagnose_latent_space(model, n_samples=20)

    # テスト条件
    pitch = 60
    pitch_norm = (pitch - 36.0) / 35.0
    cond = (0, 0, 1)  # pluck

    print("\n" + "=" * 60)
    print(f"生成テスト: pitch={pitch} (MIDI), cond={cond}")
    print("=" * 60)

    if is_collapsed:
        print("\n⚠️  Posterior Collapse検出！")
        print("z=0 での生成のみ実行します\n")

        # 方法1: z=0
        inference_with_zero_z(
            pitch_norm, cond, model, "output_method1_zero.wav"
        )

    else:
        print("\n✓ 複数の方法で生成を試します\n")

        # 方法1: z=0（ベースライン）
        inference_with_zero_z(
            pitch_norm, cond, model, "output_method1_zero.wav"
        )

        # 方法2: 学習済み分布からサンプリング
        inference_with_learned_distribution(
            pitch_norm,
            cond,
            model,
            temperature=temperature,
            output_name="output_method2_sampled.wav",
        )

        # 方法3: 小さいtemperature
        inference_with_learned_distribution(
            pitch_norm,
            cond,
            model,
            temperature=temperature * 0.5,
            output_name="output_method3_lowtemp.wav",
        )

    # 方法4: 参照音声ベース（常に実行）
    inference_from_reference(
        pitch_norm, cond, model, output_name="output_method4_reference.wav"
    )

    print("\n" + "=" * 60)
    print("生成完了！")
    print("複数の方法で生成されたwavファイルを聴き比べてください")
    print("=" * 60)

    # 異なる音高でもテスト
    print("\n追加テスト: 異なる音高で生成")
    for test_pitch in [48, 60, 72]:
        pitch_norm = (test_pitch - 36.0) / 35.0
        if is_collapsed:
            inference_with_zero_z(
                pitch_norm, cond, model, f"test_pitch_{test_pitch}_zero.wav"
            )
        else:
            inference_from_reference(
                pitch_norm,
                cond,
                model,
                output_name=f"test_pitch_{test_pitch}_ref.wav",
            )


if __name__ == "__main__":
    main()
