import tensorflow as tf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from model import TimeWiseCVAE, TIME_LENGTH, LATENT_STEPS, LATENT_DIM


def create_dummy_dataset(n_samples=100):
    """
    テスト用の合成データセット
    単純な正弦波を生成して、モデルが基本的な音を学習できるか確認
    """
    dataset = []

    for i in range(n_samples):
        # ランダムな音高
        midi = np.random.randint(36, 72)
        pitch_norm = (midi - 36.0) / 35.0

        # 周波数計算
        freq = 440.0 * 2 ** ((midi - 69) / 12.0)

        # ランダムな音色（screech, acid, pluck）
        timbre = np.random.dirichlet([1, 1, 1])

        # 条件ベクトル
        cond = np.array([pitch_norm, *timbre], dtype=np.float32)

        # 正弦波生成（基本周波数 + 倍音）
        t = np.arange(TIME_LENGTH) / 48000.0

        # 基本波
        wave = 0.5 * np.sin(2 * np.pi * freq * t)

        # 倍音を追加（音色によって変化）
        wave += 0.3 * timbre[0] * np.sin(2 * np.pi * freq * 2 * t)  # 2倍音
        wave += 0.2 * timbre[1] * np.sin(2 * np.pi * freq * 3 * t)  # 3倍音
        wave += 0.1 * timbre[2] * np.sin(2 * np.pi * freq * 4 * t)  # 4倍音

        # エンベロープ（Attack-Decay）
        attack = np.linspace(0, 1, 4800)  # 0.1秒
        decay = np.exp(-np.arange(TIME_LENGTH - 4800) / 12000)  # 減衰
        envelope = np.concatenate([attack, decay])

        wave = wave * envelope

        # 正規化
        wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.9

        dataset.append((wave[:, None].astype(np.float32), cond))

    return dataset


def visualize_waveform_and_spectrum(wave, sr=48000, title="Waveform"):
    """
    波形とスペクトログラムを可視化
    """
    from scipy import signal

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # 波形
    t = np.arange(len(wave)) / sr
    axes[0].plot(t[:4800], wave[:4800])  # 最初の0.1秒
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{title} - Waveform")
    axes[0].grid(True)

    # スペクトログラム
    f, t_spec, Sxx = signal.spectrogram(wave, sr, nperseg=2048)
    axes[1].pcolormesh(
        t_spec,
        f[:500],
        10 * np.log10(Sxx[:500] + 1e-10),
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title(f"{title} - Spectrogram")
    axes[1].set_ylim([0, 2000])

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    print(f"✓ 保存: {title}.png")


def test_reconstruction(model, test_wave, test_cond):
    """
    再構成テスト: 入力を正しく復元できるか
    """
    x_in = test_wave[None, :, :]  # (1, T, 1)
    cond_in = test_cond[None, :]  # (1, 4)

    x_hat, z_mean, z_logvar = model([x_in, cond_in])
    x_hat = tf.squeeze(x_hat).numpy()

    # 元の波形と再構成を可視化
    visualize_waveform_and_spectrum(test_wave.squeeze(), title="Original")
    visualize_waveform_and_spectrum(x_hat, title="Reconstructed")

    # MSE計算
    mse = np.mean((test_wave.squeeze() - x_hat) ** 2)
    print(f"再構成MSE: {mse:.6f}")

    # 潜在変数の統計
    z_std = np.std(z_mean.numpy())
    print(f"z_mean の標準偏差: {z_std:.6f}")
    if z_std < 0.01:
        print("⚠️  WARNING: Posterior Collapse の可能性あり")

    return x_hat


def test_generation(model, pitch=60, cond=(0, 0, 1)):
    """
    ランダム生成テスト
    """
    from model import generate_frequency_features

    pitch_norm = (pitch - 36.0) / 35.0
    cond_vec = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # ランダムな潜在変数
    z = tf.random.normal((1, LATENT_STEPS, LATENT_DIM), stddev=0.7)

    # 周波数特徴
    freq_feat = generate_frequency_features(cond_vec[:, 0], TIME_LENGTH)

    x_hat = model.decoder([z, cond_vec, freq_feat])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    x_hat = x_hat / (np.max(np.abs(x_hat)) + 1e-8) * 0.95

    visualize_waveform_and_spectrum(x_hat, title=f"Generated_pitch{pitch}")
    sf.write(f"generated_p{pitch}.wav", x_hat, samplerate=48000)

    return x_hat


def main():
    print("=" * 60)
    print("デバッグ学習スクリプト")
    print("=" * 60)

    # 1. ダミーデータ作成
    print("\n[1] テストデータ生成中...")
    dataset = create_dummy_dataset(n_samples=200)
    print(f"✓ {len(dataset)}サンプル生成完了")

    # 最初のサンプルを可視化
    visualize_waveform_and_spectrum(
        dataset[0][0].squeeze(), title="Training_Sample"
    )

    # 2. モデル構築
    print("\n[2] モデル構築中...")
    model = TimeWiseCVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # 3. 短時間学習（オーバーフィット確認）
    print("\n[3] オーバーフィット学習中...")
    print("（小さいデータセットで完全に学習できるか確認）")

    # 最初の10サンプルだけで学習
    small_data = dataset[:10]
    waves = np.stack([d[0] for d in small_data])
    conds = np.stack([d[1] for d in small_data])

    for epoch in range(50):
        history = model.fit([waves, conds], batch_size=2, epochs=1, verbose=0)

        if epoch % 10 == 0:
            metrics = history.history
            print(
                f"Epoch {epoch:3d}: loss={metrics['loss'][0]:.4f}, "
                f"recon={metrics['recon'][0]:.4f}, "
                f"z_std={metrics['z_std'][0]:.4f}"
            )

    # 4. 再構成テスト
    print("\n[4] 再構成テスト...")
    test_reconstruction(model, small_data[0][0], small_data[0][1])

    # 5. 生成テスト
    print("\n[5] 生成テスト...")
    for pitch in [48, 60, 72]:
        test_generation(model, pitch=pitch, cond=(0, 0, 1))

    print("\n" + "=" * 60)
    print("デバッグ完了！")
    print("生成された画像とwavファイルを確認してください")
    print("=" * 60)


if __name__ == "__main__":
    main()
