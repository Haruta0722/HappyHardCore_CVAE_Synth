import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from model import SR

MAX_LEN = int(SR * 1.3)
# wav loader（既存のものを想定）
def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # RMS normalize to preserve relative amplitude (avoid centering to zero mean causing trivial zero solution)
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.1  # 小さめに揃える（調整可）
    y = y / rms * target_rms
    return y.astype(np.float32)



def crop_or_pad(wav, target_len=MAX_LEN):
    length = tf.shape(wav)[0]

    def crop():
        start = tf.random.uniform(
            (), 0, length - target_len + 1, dtype=tf.int32
        )
        return wav[start : start + target_len]

    def pad():
        pad_len = target_len - length
        return tf.pad(wav, [[0, pad_len]], constant_values=0.0)

    return tf.cond(length > target_len, crop, pad)


# -------------------------
# Dataset
# -------------------------
def make_dataset_from_synth_csv(
    csv_path,
    base_dir=".",
    batch_size=16,
    shuffle=True,
):
    df = pd.read_csv(csv_path)

    # 絶対パス化
    df["path"] = df["path"].apply(
        lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p
    )

    def gen():
        for _, row in df.iterrows():
            x = load_wav(row["path"])

            # pitch: 36–71 → 0–1 正規化
            pitch = (row["pitch"] - 36.0) / (71.0 - 36.0)

            cond = np.array(
                [
                    pitch,
                    row["screech"],
                    row["acid"],
                    row["pluck"],
                ],
                dtype=np.float32,
            )


            # autoencoder なので x=y
            yield x, cond

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(256)

    ds = ds.map(
        lambda x, c: (
            tf.expand_dims(crop_or_pad(x), -1),
            c,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
