from model import build_decoder, build_encoder, TimeWiseCVAE
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf
import os


def train_model():
    dataset = make_dataset_from_synth_csv("dataset_filtered.csv", batch_size=16)
    model = TimeWiseCVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/epoch_{epoch:03d}",
        save_weights_only=False,   # True にすると軽量（おすすめ）
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