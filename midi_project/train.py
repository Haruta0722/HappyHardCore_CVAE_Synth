from model import build_decoder, build_encoder, TimeWiseCVAE
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf
import os


def train_model():
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=16)
    model = TimeWiseCVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5))

    os.makedirs("checkpoints", exist_ok=True)

    for ds in dataset.take(1):
        model.build(ds)
        print("モデルの入力形状を構築しました。")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/epoch_{epoch:03d}.weights.h5",
        save_weights_only=True,  # True にすると軽量（おすすめ）
        save_freq="epoch",
    )
    model.fit(
        dataset,
        epochs=100,
        callbacks=[checkpoint_cb],
    )


if __name__ == "__main__":
    print("Starting training...")
    train_model()
    print("Training completed.")
