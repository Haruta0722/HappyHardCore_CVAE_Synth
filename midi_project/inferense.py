import soundfile as sf
import tensorflow as tf
import numpy as np
from model import TimeWiseCVAE, LATENT_DIM, LATENT_STEPS


def inference(pitch: float, cond: tuple[float, float, float], temperature=0.8):
    """
    音声を生成

    Args:
        pitch: MIDI音高 (36-71の範囲)
        cond: (screech, acid, pluck) の3つ組
        temperature: 生成のランダム性 (0.5-1.5推奨)
    """
    # モデル読み込み
    model = TimeWiseCVAE()
    model.build(
        [
            (None, 62400, 1),  # TIME_LENGTHを明示
            (None, 4),
        ]
    )

    ckpt_path = "weights/epoch_100.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデルの重みを読み込みました: {ckpt_path}")

    # 条件ベクトル作成
    pitch_normalized = (pitch - 36.0) / (71.0 - 36.0)
    cond_vector = tf.constant([[pitch_normalized, *cond]], dtype=tf.float32)

    # ★改善7: 正しいステップ数でzを生成
    # temperatureで生成のランダム性を調整
    z = tf.random.normal(
        shape=(1, LATENT_STEPS, LATENT_DIM), stddev=temperature
    )

    # デコード
    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat, axis=0).numpy()

    # 正規化して保存
    x_hat = x_hat / (np.max(np.abs(x_hat)) + 1e-8) * 0.95
    sf.write("generated_output.wav", x_hat, samplerate=48000)

    print(f"✓ 生成完了: pitch={pitch:.1f}, cond={cond}, temp={temperature}")
    return x_hat


def inference_from_latent_mean(pitch: float, cond: tuple[float, float, float]):
    """
    ランダムサンプリングではなく、潜在空間の平均値を使って生成
    より安定した出力が得られる
    """
    model = TimeWiseCVAE()
    model.build(
        [
            (None, 62400, 1),
            (None, 4),
        ]
    )

    ckpt_path = "weights/epoch_100.weights.h5"
    model.load_weights(ckpt_path)

    pitch_normalized = (pitch - 36.0) / (71.0 - 36.0)
    cond_vector = tf.constant([[pitch_normalized, *cond]], dtype=tf.float32)

    # ★改善8: ゼロ平均の潜在ベクトルを使用
    z = tf.zeros(shape=(1, LATENT_STEPS, LATENT_DIM), dtype=tf.float32)

    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat, axis=0).numpy()
    x_hat = x_hat / (np.max(np.abs(x_hat)) + 1e-8) * 0.95

    sf.write("generated_output_mean.wav", x_hat, samplerate=48000)
    print(f"✓ 平均値で生成完了: pitch={pitch:.1f}, cond={cond}")
    return x_hat


if __name__ == "__main__":
    # テスト1: ランダムサンプリング
    pitch = 60  # C4
    cond = (0, 0, 1)  # pluckのみ
    inference(pitch, cond, temperature=0.7)

    # テスト2: 平均値での生成（より安定）
    inference_from_latent_mean(pitch, cond)

    # テスト3: 異なる音高
    for p in [
        48,
        60,
    ]:  # C3, C4, C5
        inference(p, (0, 0, 1), temperature=0.7)
        sf.write(
            f"generated_pitch_{p}.wav",
            inference(p, cond, temperature=0.7),
            samplerate=48000,
        )

    print("\n✓ すべての生成が完了しました")
