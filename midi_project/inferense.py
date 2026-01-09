import soundfile as sf
import tensorflow as tf
import numpy as np
from model import TimeWiseCVAE, LATENT_STEPS, LATENT_DIM, TIME_LENGTH
from create_datasets import load_wav, crop_or_pad


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


def inference_from_reference(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    reference_wave=None,
    output_name="generated_ref.wav",
):
    """
    ★改善版: 参照音声から潜在変数を抽出して生成
    エンベロープの時間変化もzに反映されるようになった
    """
    print(f"\n[生成] 参照ベース（改善版）")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    if reference_wave is None:
        t = np.arange(TIME_LENGTH) / 48000.0
        freq = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
        reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
        reference_wave = reference_wave[:, None].astype(np.float32)
    else:
        reference_wave = np.array(reference_wave, dtype=np.float32)
        if reference_wave.ndim == 1:
            reference_wave = reference_wave[:, None]

    # 条件ベクトル
    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # 参照からzを抽出
    reference_wave_batch = tf.constant(
        reference_wave[None, :, :], dtype=tf.float32
    )
    z_mean, z_logvar = model.encoder([reference_wave_batch, cond_vector])

    # ★重要: 平均値だけでなく、適度なランダム性も加える
    # これにより、エンベロープの微妙な変化も再現される
    z = (
        z_mean
        + tf.exp(0.5 * z_logvar) * tf.random.normal(tf.shape(z_mean)) * 0.1
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


def inference_random_z(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    temperature=0.8,
    output_name="generated.wav",
):
    """ランダムな潜在変数から生成"""
    print(f"\n[生成] ランダムz (temperature={temperature:.2f})")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    z = tf.keras.random.normal(
        (1, LATENT_STEPS, LATENT_DIM), stddev=temperature
    )

    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"  ✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def test_envelope_learning(model, reference_files, output_dir="envelope_test"):
    """
    ★新機能: エンベロープ学習のテスト
    異なる参照音を使って、エンベロープが正しく再現されるか確認
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("エンベロープ学習テスト")
    print("=" * 60)

    for ref_file in reference_files:
        print(f"\n参照: {ref_file}")

        # 参照音声を読み込み
        reference = load_wav(ref_file)
        reference = crop_or_pad(reference, TIME_LENGTH)

        # 音高を推定（簡易版）
        # 実際にはより正確な音高推定が必要
        pitch = 60  # C4

        # 各音色で生成
        for timbre_name, cond in [
            ("pluck", (0, 0, 1)),
            ("screech", (1, 0, 0)),
            ("acid", (0, 1, 0)),
        ]:
            output_name = os.path.join(
                output_dir,
                f"{os.path.basename(ref_file).replace('.wav', 'epoch_137e_name}.wav')}",
            )

            inference_from_reference(
                pitch,
                cond,
                model,
                reference_wave=reference,
                output_name=output_name,
            )

    print("\n✓ エンベロープテスト完了")
    print(f"  結果: {output_dir}/")


def compare_envelope_shapes(model, pitch=60, output_dir="envelope_comparison"):
    """
    ★新機能: 異なる音色のエンベロープ形状を比較
    acidのうねり（LFO変調）も可視化
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("エンベロープ形状比較")
    print("=" * 60)

    timbre_configs = {
        "screech": (1, 0, 0),
        "acid": (0, 1, 0),
        "pluck": (0, 0, 1),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (timbre_name, cond) in enumerate(timbre_configs.items()):
        print(f"\n{timbre_name}...")

        # 生成
        audio = inference_zero_z(
            pitch,
            cond,
            model,
            output_name=f"{output_dir}/{timbre_name}.wav",
        )

        # エンベロープ（振幅包絡）を抽出
        hop_size = 512
        envelope = np.array(
            [
                np.max(np.abs(audio[i : i + hop_size]))
                for i in range(0, len(audio) - hop_size, hop_size)
            ]
        )

        time_axis = np.arange(len(envelope)) * hop_size / 48000.0

        # 上段: エンベロープ
        ax1 = axes[0, idx]
        ax1.plot(time_axis, envelope)
        ax1.set_title(f"{timbre_name} - Envelope")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.set_ylim([0, 1.0])

        # 下段: スペクトログラム（周波数特性を確認）
        ax2 = axes[1, idx]
        import librosa
        import librosa.display

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)), ref=np.max
        )
        librosa.display.specshow(
            D,
            sr=48000,
            hop_length=512,
            x_axis="time",
            y_axis="hz",
            ax=ax2,
            cmap="viridis",
        )
        ax2.set_title(f"{timbre_name} - Spectrogram")
        ax2.set_ylim([0, 8000])  # 8kHzまで表示

        # acidの場合、うねりを強調表示
        if timbre_name == "acid":
            # エンベロープの細かい変動を検出
            from scipy.signal import hilbert

            analytic_signal = hilbert(envelope - np.mean(envelope))
            amplitude_envelope = np.abs(analytic_signal)
            ax1.plot(
                time_axis,
                np.mean(envelope) + amplitude_envelope * 0.1,
                "r--",
                alpha=0.5,
                label="LFO modulation",
            )
            ax1.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/envelope_comparison.png", dpi=150)
    print(f"\n✓ エンベロープ比較図を保存: {output_dir}/envelope_comparison.png")
    plt.close()


def diagnose_model(model):
    """モデルの状態を診断"""
    print("\n" + "=" * 60)
    print("モデル診断")
    print("=" * 60)

    dummy_waves = tf.keras.random.normal((20, TIME_LENGTH, 1))
    cond_dim = tf.keras.random.normal((1, 4))
    z_means = []
    z_logvars = []

    for i in range(20):
        wave = dummy_waves[i : i + 1]
        z_mean, z_logvar = model.encoder([wave, cond_dim])
        z_means.append(z_mean.numpy())
        z_logvars.append(z_logvar.numpy())

    z_means = np.concatenate(z_means, axis=0)
    z_logvars = np.concatenate(z_logvars, axis=0)

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
    reference = load_wav("datasets/C3/0013.wav")
    reference = crop_or_pad(reference, TIME_LENGTH)
    print("=" * 60)
    print("DDSP風モデル 推論スクリプト（改善版）")
    print("=" * 60)

    print("\n[1] モデル読み込み中...")
    model = TimeWiseCVAE()

    dummy_x = tf.zeros((1, TIME_LENGTH, 1))
    dummy_cond = tf.zeros((1, 4))
    _ = model((dummy_x, dummy_cond), training=False)

    ckpt_path = "weights/epoch_137.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデルの重みを読み込みました: {ckpt_path}")

    print("\n[2] モデル診断中...")
    recommended_temp = diagnose_model(model)

    print("\n[3] 基本テスト生成...")
    pitch = 60

    # 各音色でテスト
    for timbre_name, cond in [
        ("pluck", (0, 0, 1)),
        ("screech", (1, 0, 0)),
        ("acid", (0, 1, 0)),
    ]:
        print(f"\n--- {timbre_name} ---")

        # ランダム生成
        inference_random_z(
            pitch,
            cond,
            model,
            temperature=recommended_temp,
            output_name=f"test_{timbre_name}_random.wav",
        )

        # 参照ベース（正弦波から）
        inference_from_reference(
            pitch,
            cond,
            model,
            reference_wave=reference,
            output_name=f"test_{timbre_name}_ref.wav",
        )

    print("\n[4] エンベロープ形状比較...")
    compare_envelope_shapes(model, pitch=60)

    print("\n[5] 参照音声を使ったテスト...")
    # あなたのデータセットから実際の音声ファイルを使用
    # reference_files = [
    #     "datasets/C4/pluck_001.wav",
    #     "datasets/C4/screech_001.wav",
    #     "datasets/C4/acid_001.wav",
    # ]
    # test_envelope_learning(model, reference_files)

    print("\n" + "=" * 60)
    print("推論完了！")
    print("=" * 60)
    print("\n確認項目:")
    print("  1. pluckは急速な減衰（アタック感が強い）を示しているか？")
    print("  2. screechは持続的なノイズと高音域の倍音があるか？")
    print("  3. acidは中音域のうねり（LFO変調）が聞こえるか？")
    print("  4. 3つの音色は互いに独立した特徴を持っているか？")
    print("  5. envelope_comparison.png でエンベロープの違いを確認")
    print("     - acidはうねりによる振幅変調が見えるはず")
    print("=" * 60)


if __name__ == "__main__":
    main()
