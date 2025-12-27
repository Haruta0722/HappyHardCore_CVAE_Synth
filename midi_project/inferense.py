import soundfile as sf
import tensorflow as tf
import numpy as np
from model import TimeWiseCVAE, LATENT_STEPS, LATENT_DIM, TIME_LENGTH
from create_datasets import load_wav, crop_or_pad


def inference_random_z(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    temperature=0.8,
    output_name="generated.wav",
):
    """
    ランダムな潜在変数から生成

    Args:
        pitch: MIDI音高 (36-71)
        cond: (screech, acid, pluck) の3つ組
        model: 学習済みモデル
        temperature: ランダム性の強さ
        output_name: 出力ファイル名
    """
    print(f"\n[生成] ランダムz (temperature={temperature:.2f})")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    # 条件ベクトル
    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # ランダムな潜在変数
    z = tf.keras.random.normal(
        (1, LATENT_STEPS, LATENT_DIM), stddev=temperature
    )

    # 生成
    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"  ✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def inference_from_reference(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    reference_wave=None,
    output_name="generated_ref.wav",
):
    """
    参照音声から潜在変数を抽出して生成
    最も安定した方法

    Args:
        pitch: MIDI音高
        cond: (screech, acid, pluck)
        model: 学習済みモデル
        reference_wave: 参照波形 (None の場合は正弦波を生成)
        output_name: 出力ファイル名
    """
    print(f"\n[生成] 参照ベース")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    if reference_wave is None:
        # シンプルな正弦波を生成
        t = np.arange(TIME_LENGTH) / 48000.0
        freq = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
        reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
        reference_wave = reference_wave[:, None].astype(np.float32)
    else:
        # ★ここに追加: 外部から渡された波形が1次元の場合、(Time, 1)の形状に変換する
        reference_wave = np.array(reference_wave, dtype=np.float32)  # 型を保証
        if reference_wave.ndim == 1:
            reference_wave = reference_wave[:, None]

    # 条件ベクトル
    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # 参照からzを抽出
    # ここでの [None, :, :] は reference_wave が2次元 (Time, Channels) であることを前提としています
    reference_wave_batch = tf.constant(
        reference_wave[None, :, :], dtype=tf.float32
    )
    z_mean, z_logvar = model.encoder([reference_wave_batch, cond_vector])
    z = z_mean  # 平均値を使用

    # 生成
    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"  ✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def inference_zero_z(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    output_name="generated_zero.wav",
):
    """
    z=0 で生成（デバッグ用）
    条件ベクトルだけで生成
    """
    print(f"\n[生成] z=0 (条件のみ)")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    # 条件ベクトル
    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # ゼロの潜在変数
    z = tf.zeros((1, LATENT_STEPS, LATENT_DIM), dtype=tf.float32)

    # 生成
    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"  ✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def test_pitch_range(model, cond=(0, 0, 1), output_dir="test_outputs"):
    """
    異なる音高でテスト生成
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("音高範囲テスト")
    print("=" * 60)

    # C2からC6まで1オクターブごと
    pitches = [36, 48, 60, 72, 84]  # C2, C3, C4, C5, C6

    for pitch in pitches:
        note_name = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ][pitch % 12]
        octave = (pitch // 12) - 1

        print(f"\n{note_name}{octave} (MIDI {pitch}):")

        # ランダム生成
        inference_random_z(
            pitch,
            cond,
            model,
            temperature=0.7,
            output_name=f"{output_dir}/pitch_{pitch:02d}_random.wav",
        )

        # z=0で生成
        inference_zero_z(
            pitch,
            cond,
            model,
            output_name=f"{output_dir}/pitch_{pitch:02d}_zero.wav",
        )

    print("\n✓ 音高テスト完了")


def test_timbre_variations(model, pitch=60, output_dir="test_outputs"):
    """
    異なる音色でテスト生成
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("音色バリエーションテスト")
    print("=" * 60)

    conditions = {
        "screech": (1, 0, 0),
        "acid": (0, 1, 0),
        "pluck": (0, 0, 1),
        "screech_acid": (0.5, 0.5, 0),
        "acid_pluck": (0, 0.5, 0.5),
        "all_mix": (0.33, 0.33, 0.34),
    }

    for name, cond in conditions.items():
        print(f"\n{name}: {cond}")

        # ランダム生成
        inference_random_z(
            pitch,
            cond,
            model,
            temperature=0.7,
            output_name=f"{output_dir}/timbre_{name}_random.wav",
        )

        # z=0で生成
        inference_zero_z(
            pitch,
            cond,
            model,
            output_name=f"{output_dir}/timbre_{name}_zero.wav",
        )

    print("\n✓ 音色テスト完了")


def diagnose_model(model):
    """
    モデルの状態を診断
    """
    print("\n" + "=" * 60)
    print("モデル診断")
    print("=" * 60)

    # ダミー入力で潜在変数の統計を確認
    dummy_waves = tf.keras.random.normal((20, TIME_LENGTH, 1))
    cond_dim = tf.keras.random.normal((1, 4))  # ダミー条件ベクトル
    z_means = []
    z_logvars = []

    for i in range(20):
        wave = dummy_waves[i : i + 1]
        z_mean, z_logvar = model.encoder([wave, cond_dim])
        z_means.append(z_mean.numpy())
        z_logvars.append(z_logvar.numpy())

    z_means = np.concatenate(z_means, axis=0)
    z_logvars = np.concatenate(z_logvars, axis=0)

    # 統計
    mean_of_means = np.mean(z_means)
    std_of_means = np.std(z_means)
    mean_of_logvars = np.mean(z_logvars)

    print(f"\n潜在変数の統計:")
    print(f"  z_mean の平均: {mean_of_means:.6f}")
    print(f"  z_mean の標準偏差: {std_of_means:.6f}")
    print(f"  z_logvar の平均: {mean_of_logvars:.6f}")

    print("\n診断結果:")
    if std_of_means < 0.01:
        print("⚠️  WARNING: Posterior Collapse の可能性")
        print("   推論時は z=0 または参照ベースを推奨")
    else:
        print("✓ 潜在変数は適切に活用されています")
        print(f"   推論時の推奨 temperature: {std_of_means:.3f}")

    print("=" * 60)

    return std_of_means


def main():
    print("=" * 60)
    print("DDSP風モデル 推論スクリプト")
    print("=" * 60)

    # モデル読み込み
    print("\n[1] モデル読み込み中...")
    model = TimeWiseCVAE()

    # ダミーデータでビルド
    dummy_x = tf.zeros((1, TIME_LENGTH, 1))
    dummy_cond = tf.zeros((1, 4))
    _ = model((dummy_x, dummy_cond), training=False)

    ckpt_path = "weights/epoch_200.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデルの重みを読み込みました: {ckpt_path}")

    # モデル診断
    print("\n[2] モデル診断中...")
    recommended_temp = diagnose_model(model)

    # テスト生成
    print("\n[3] テスト生成中...")

    # 基本テスト
    pitch = 60  # C4
    cond = (0, 0, 1)  # pluck

    inference_random_z(
        pitch,
        cond,
        model,
        temperature=recommended_temp,
        output_name="test_random.wav",
    )
    inference_zero_z(pitch, cond, model, output_name="test_zero.wav")
    reference = load_wav("datasets/C4/0013.wav")
    reference = crop_or_pad(reference, TIME_LENGTH)
    inference_from_reference(
        pitch,
        cond,
        model,
        reference_wave=reference,
        output_name="test_reference.wav",
    )

    # 詳細テスト
    print("\n[4] 詳細テスト中...")
    test_pitch_range(model, cond=(0, 0, 1))
    test_timbre_variations(model, pitch=60)

    print("\n" + "=" * 60)
    print("推論完了！")
    print("=" * 60)
    print("\n生成されたファイル:")
    print("  test_random.wav - ランダム生成")
    print("  test_zero.wav - z=0 生成")
    print("  test_reference.wav - 参照ベース生成")
    print("  test_outputs/ - 詳細テスト結果")
    print("\n推奨:")
    print("  - 音高が正しく出ているか確認")
    print("  - 異なる音色で違いが出ているか確認")
    print("  - スペクトログラムで倍音構造を確認")
    print("=" * 60)


if __name__ == "__main__":
    main()
