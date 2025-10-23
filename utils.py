import tensorflow as tf
import os

SR = 32000  # サンプルレート
FRAME_LENGTH = 1024
FRAME_STEP = 256


# --- 音声ロード関数 ---
def load_wav(path):
    audio_binary = tf.io.read_file(path)
    decoded = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(decoded, axis=-1)
    return audio


# --- STFT変換 ---
def preprocess_audio(audio):
    # 長さが違ってもSTFTは処理できる
    stft = tf.signal.stft(
        audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP
    )
    mag = tf.abs(stft)
    log_mag = tf.math.log(mag + 1e-6)
    return log_mag


# --- Dataset構築 ---
def build_dataset(audio_dir):
    file_list = tf.io.gfile.glob(os.path.join(audio_dir, "*.wav"))
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    # ファイル → 波形
    dataset = dataset.map(
        lambda x: load_wav(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    # STFTなどの前処理
    dataset = dataset.map(
        lambda x: preprocess_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )

    # バッチ化（任意長に対応するためパディングバッチ）
    dataset = dataset.padded_batch(8, padded_shapes=[None, None])

    return dataset.prefetch(tf.data.AUTOTUNE)
