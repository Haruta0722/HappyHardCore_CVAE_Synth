import soundfile as sf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import TimeWiseCVAE, LATENT_DIM, LATENT_STEPS, TIME_LENGTH


def diagnose_model(ckpt_path="weights/epoch_100.weights.h5"):
    """
    モデルの状態を診断
    潜在変数が使われているかを確認
    """
    print("=" * 60)
    print("モデル診断")
    print("=" * 60)

    model = TimeWiseCVAE()
    model.build(
        [
            (None, TIME_LENGTH, 1),
            (None, 4),
        ]
    )
    model.load_weights(ckpt_path)

    # テストデータで潜在変数の分散を確認
    test_pitches = [48, 60, 72]  # C3, C4, C5

    for pitch in test_pitches:
        pitch_norm = (pitch - 36.0) / 35.0
        cond = tf.constant([[pitch_norm, 0, 0, 1]], dtype=tf.float32)

        # ダミー入力（ノイズ）
        dummy_x = tf.random.normal((1, TIME_LENGTH, 1))

        z_mean, z_logvar = model.encoder([dummy_x, cond])

        mean_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1)).numpy()
        logvar_mean = tf.reduce_mean(z_logvar).numpy()

        print(
            f"Pitch {pitch:2d}: z_mean_std={mean_std:.4f}, z_logvar_mean={logvar_mean:.4f}"
        )

    print("\n診断結果:")
    if mean_std < 0.01:
        print("⚠️  WARNING: 潜在変数の分散が極端に小さい（Posterior Collapse）")
    elif mean_std > 2.0:
        print("⚠️  WARNING: 潜在変数の分散が大きすぎる（学習不安定）")
    else:
        print("✓ 潜在変数は適切に使われています")

    print("=" * 60)


def inference(
    pitch: float,
    cond: tuple[float, float, float],
    ckpt_path="weights/epoch_100.weights.h5",
    use_mean=False,
    temperature=0.8,
):
    """
    音声生成

    Args:
        pitch: MIDI音高 (36-71)
        cond: (screech, acid, pluck)
        use_mean: Trueの場合、zの平均値(ゼロ)を使用
        temperature: ランダム性の強さ
    """
    model = TimeWiseCVAE()
    model.build(
        [
            (None, TIME_LENGTH, 1),
            (None, 4),
        ]
    )
    model.load_weights(ckpt_path)

    pitch_normalized = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_normalized, *cond]], dtype=tf.float32)

    if use_mean:
        # 条件ベクトルだけで生成（デバッグ用）
        z = tf.zeros((1, LATENT_STEPS, LATENT_DIM))
        suffix = "_mean"
    else:
        z = tf.random.normal((1, LATENT_STEPS, LATENT_DIM), stddev=temperature)
        suffix = f"_temp{temperature}"

    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    filename = (
        f"gen_p{int(pitch)}_{''.join(map(str, map(int, cond)))}{suffix}.wav"
    )
    sf.write(filename, x_hat, samplerate=48000)

    print(f"✓ 生成: {filename} (max_amp={max_val:.4f})")
    return x_hat, filename


def test_pitch_response(ckpt_path="weights/epoch_100.weights.h5"):
    """
    音高応答テスト: 異なる音高で生成して比較
    """
    print("\n" + "=" * 60)
    print("音高応答テスト")
    print("=" * 60)

    pitches = [36, 48, 60, 72]  # C2, C3, C4, C5
    cond = (0, 0, 1)  # pluck

    waveforms = []

    for pitch in pitches:
        wav, filename = inference(pitch, cond, ckpt_path, use_mean=True)
        waveforms.append(wav)

        # 簡易的な基本周波数推定（ゼロ交差）
        zero_crossings = np.where(np.diff(np.sign(wav)))[0]
        if len(zero_crossings) > 1:
            avg_period = np.mean(np.diff(zero_crossings))
            estimated_freq = 48000 / (2 * avg_period)
            expected_freq = 440 * 2 ** ((pitch - 69) / 12)
            print(
                f"  期待周波数: {expected_freq:.1f} Hz, 推定: {estimated_freq:.1f} Hz"
            )

    print("=" * 60)
    return waveforms


def visualize_spectrum(wav, sr=48000, title="Spectrum"):
    """
    スペクトル可視化
    """
    from scipy import signal

    f, t, Sxx = signal.spectrogram(wav, sr, nperseg=2048)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(
        t, f[:1000], 10 * np.log10(Sxx[:1000] + 1e-10), shading="auto"
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"✓ スペクトログラムを保存: {title}.png")


if __name__ == "__main__":
    ckpt = "weights/epoch_100.weights.h5"

    # 1. モデル診断
    diagnose_model(ckpt)

    # 2. 音高応答テスト
    waveforms = test_pitch_response(ckpt)

    # 3. スペクトル可視化
    if len(waveforms) > 0:
        visualize_spectrum(waveforms[2], title="Pitch_60_Spectrum")

    # 4. 複数条件でテスト
    print("\n" + "=" * 60)
    print("音色バリエーションテスト")
    print("=" * 60)

    pitch = 60
    conditions = [
        (1, 0, 0),  # screech
        (0, 1, 0),  # acid
        (0, 0, 1),  # pluck
        (0.5, 0.5, 0),  # screech + acid
    ]

    for cond in conditions:
        inference(pitch, cond, ckpt, use_mean=False, temperature=0.7)

    print("\n✓ すべてのテストが完了しました")
