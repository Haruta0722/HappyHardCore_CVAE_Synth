import soundfile as sf
import tensorflow as tf
from model import TimeWiseCVAE, LATENT_DIM, generate_frequency_features
from create_datasets import MAX_LEN


def inferense(pitch: float, cond: tuple[float, float, float]):
    # モデル読み込み
    model = TimeWiseCVAE()
    model.build(
        [  # 入力形状を指定
            (None, MAX_LEN, 1),  # 波形 x
            (None, 4),  # 条件ベクトル cond
        ]
    )
    ckpt_path = "weights/epoch_100.weights.h5"
    model.load_weights(ckpt_path)
    print(f"モデルの重みを {ckpt_path} から読み込みました。")

    # 条件ベクトル作成

    cond_vector = tf.constant(
        [[pitch, *cond]], dtype=tf.float32
    )  # バッチサイズ1
    z = tf.random.normal(shape=(1, MAX_LEN // 16, LATENT_DIM))
    z = tf.convert_to_tensor(z)
    # 周波数特徴を生成
    freq_feat = generate_frequency_features(pitch, MAX_LEN)
    x_hat = model.decoder([z, cond_vector, freq_feat])
    x_hat = tf.squeeze(x_hat, axis=0).numpy()  # [T, 1] -> [T]
    sf.write("generated_output.wav", x_hat, samplerate=48000)


if __name__ == "__main__":
    pitch = 60
    pitch = (pitch - 36.0) / (71.0 - 36.0)
    cond = (0, 0, 1)  # screech, acid, pluck
    inferense(pitch, cond)
    print("生成された音声を 'generated_output.wav' に保存しました。")
