from model import TimeWiseCVAE
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf
import os


def train_model(resume_checkpoint=True):
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=16)
    model = TimeWiseCVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5))

    checkpoint_path = "checkpoints/best_model.weights.h5"
    initial_epoch = 0
    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"🔄  チェックポイントを発見: {checkpoint_path} から学習再開")
        model.load_weights(checkpoint_path)
        # CSVログから最後のepochを取得して再開epochを設定
        import csv

        if os.path.exists("training_log.csv"):
            with open("training_log.csv", "r") as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    last_epoch = int(reader[-1][0])
                    initial_epoch = last_epoch + 1
                    print(f"  CSVログより初期エポックを {initial_epoch} に設定")
    else:
        print("🆕  新規学習を開始します")

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
        steps_per_epoch=87,
    )


if __name__ == "__main__":
    print("Starting training...")
    train_model()
    print("Training completed.")
