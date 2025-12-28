import tensorflow as tf
import numpy as np
import os
from model import TimeWiseCVAE, TIME_LENGTH
from create_datasets import (
    make_dataset_from_synth_csv,
)  # あなたのデータセット作成関数

# GPUメモリ設定
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def create_callbacks(save_dir="weights"):
    """訓練用コールバック"""
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        # チェックポイント保存（10エポックごと）
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "epoch_{epoch:03d}.weights.h5"),
            save_weights_only=True,
            save_freq="epoch",
            period=10,
            verbose=1,
        ),
        # 最良モデル保存
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
        # 学習率スケジューリング
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1
        ),
        # Early Stopping（オプション）
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=50, restore_best_weights=True, verbose=1
        ),
        # カスタムコールバック：学習状況の詳細表示
        DetailedLogger(),
    ]

    return callbacks


class DetailedLogger(tf.keras.callbacks.Callback):
    """詳細なログ出力"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} 完了")
        print(f"{'='*60}")
        print(f"Loss: {logs.get('loss', 0):.6f}")
        print(f"  - Recon: {logs.get('recon', 0):.6f}")
        print(f"  - STFT: {logs.get('stft', 0):.6f}")
        print(f"  - Mel: {logs.get('mel', 0):.6f}")
        print(
            f"  - KL: {logs.get('kl', 0):.6f} (weight: {logs.get('kl_weight', 0):.6f})"
        )
        print(f"Z stats:")
        print(f"  - std_ema: {logs.get('z_std_ema', 0):.6f}")
        print(f"  - grad_norm: {logs.get('grad_norm', 0):.6f}")
        print(f"{'='*60}\n")


def main():
    print("=" * 60)
    print("DDSP風モデル 訓練スクリプト")
    print("=" * 60)

    # ハイパーパラメータ
    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 1e-4

    print(f"\n設定:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # データセット作成
    print("\n[1] データセット読み込み中...")
    train_dataset = make_dataset_from_synth_csv(
        "dataset.csv",
        batch_size=BATCH_SIZE,
        # ここにあなたのデータセットパラメータを追加
    )
    train_dataset = train_dataset.repeat()

    # 1エポックあたりのステップ数を計算
    # （データセットのサイズに応じて調整してください）
    steps_per_epoch = 87  # あなたのデータセットサイズに合わせて変更

    print(f"✓ データセット読み込み完了")
    print(f"  Steps per epoch: {steps_per_epoch}")

    # モデル構築
    print("\n[2] モデル構築中...")
    model = TimeWiseCVAE(steps_per_epoch=steps_per_epoch)

    # オプティマイザ
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, clipnorm=1.0  # 勾配クリッピング
    )
    model.compile(optimizer=optimizer)

    # ダミーデータでモデルをビルド
    dummy_x = tf.zeros((1, TIME_LENGTH, 1))
    dummy_cond = tf.zeros((1, 4))
    _ = model((dummy_x, dummy_cond), training=False)

    print("✓ モデル構築完了")
    model.encoder.summary()
    print("\n")
    model.decoder.summary()

    # コールバック設定
    print("\n[3] コールバック設定...")
    callbacks = create_callbacks()

    # 訓練開始
    print("\n[4] 訓練開始")
    print("=" * 60)

    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        print("\n" + "=" * 60)
        print("訓練完了！")
        print("=" * 60)

        # 最終統計
        final_loss = history.history["loss"][-1]
        final_recon = history.history["recon"][-1]
        final_kl = history.history["kl"][-1]

        print(f"\n最終結果:")
        print(f"  Loss: {final_loss:.6f}")
        print(f"  Reconstruction: {final_recon:.6f}")
        print(f"  KL: {final_kl:.6f}")

        # 最良エポックの情報
        best_epoch = np.argmin(history.history["loss"]) + 1
        best_loss = np.min(history.history["loss"])
        print(f"\n最良エポック: {best_epoch}")
        print(f"  Loss: {best_loss:.6f}")

    except KeyboardInterrupt:
        print("\n訓練が中断されました")
        print("最後のチェックポイントが保存されています")

    print("\n保存場所:")
    print("  weights/best_model.weights.h5 - 最良モデル")
    print("  weights/epoch_XXX.weights.h5 - 各エポックのチェックポイント")
    print("  logs/ - TensorBoardログ")

    print("\n次のステップ:")
    print("  1. TensorBoardで訓練曲線を確認")
    print("     $ tensorboard --logdir=logs")
    print("  2. inference_improved.py で推論テスト")
    print("=" * 60)


if __name__ == "__main__":
    main()
