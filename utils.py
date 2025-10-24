import pandas as pd
import tensorflow as tf

SR = 32000  # サンプルレート
FRAME_LENGTH = 1024
FRAME_STEP = 256


# --- 音声ロード関数 ---
def load_wav(path):
    audio_binary = tf.io.read_file(path)
    decoded = tf.audio.decode_wav(audio_binary)
    audio_data = getattr(decoded, "audio")
    audio = tf.squeeze(audio_data, axis=-1)
    return audio


# --- STFT変換 ---
def preprocess_audio(audio):
    stft = tf.signal.stft(
        audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP
    )
    mag = tf.abs(stft)
    log_mag = tf.math.log(mag + 1e-6)
    return log_mag


# --- 1サンプルの読み込み処理 ---
def load_and_preprocess(
    input_path, output_path, attack, distortion, thickness, center_tone
):
    x = load_wav(input_path)
    y = load_wav(output_path)
    x = preprocess_audio(x)
    y = preprocess_audio(y)
    cond = tf.stack([attack, distortion, thickness, center_tone])
    return (x, cond), y


# --- Dataset構築関数 ---
def build_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    input_paths = df["input_path"].values
    output_paths = df["output_path"].values
    attacks = df["attack"].values
    distortions = df["distortion"].values
    thicknesses = df["thickness"].values
    centers = df["center_tone"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_paths, output_paths, attacks, distortions, thicknesses, centers)
    )
    dataset = dataset.map(
        lambda i, o, a, d, t, c: load_and_preprocess(i, o, a, d, t, c),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.padded_batch(
        4, padded_shapes=(((None, None), (4,)), (None, None))
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
    

