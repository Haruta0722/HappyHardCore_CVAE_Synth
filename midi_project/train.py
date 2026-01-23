from model import TimeWiseCVAE, TIME_LENGTH
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf
import os
import csv


class CSVLogger(tf.keras.callbacks.Callback):
    """CSVログを記録するカスタムコールバック"""

    def __init__(self, filename="training_log.csv"):
        super().__init__()
        self.filename = filename
        self.file = None
        self.writer = None

    def on_train_begin(self, logs=None):
        # ファイルが存在しない場合はヘッダーを書く
        file_exists = os.path.exists(self.filename)
        self.file = open(self.filename, "a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            # ヘッダー行
            self.writer.writerow(
                [
                    "epoch",
                    "loss",
                    "recon",
                    "stft",
                    "mel",
                    "kl",
                    "kl_weight",
                    "prototype_loss",
                    "z_std_ema",
                    "grad_norm",
                ]
            )
            self.file.flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # ログに記録
        self.writer.writerow(
            [
                epoch,
                logs.get("loss", 0),
                logs.get("recon", 0),
                logs.get("stft", 0),
                logs.get("mel", 0),
                logs.get("kl", 0),
                logs.get("kl_weight", 0),
                logs.get("prototype_loss", 0),
                logs.get("z_std_ema", 0),
                logs.get("grad_norm", 0),
            ]
        )
        self.file.flush()

    def on_train_end(self, logs=None):
        if self.file:
            self.file.close()


class DetailedProgressLogger(tf.keras.callbacks.Callback):
    """詳細な進捗を表示"""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1} 完了")
        print(f"{'='*70}")
        print(f"Loss: {logs.get('loss', 0):.6f}")
        print(f"  Reconstruction: {logs.get('recon', 0):.6f}")
        print(f"  STFT: {logs.get('stft', 0):.6f}")
        print(f"  Mel: {logs.get('mel', 0):.6f}")
        print(
            f"  KL: {logs.get('kl', 0):.6f} (weight: {logs.get('kl_weight', 0):.6f})"
        )
        print(f"  Prototype: {logs.get('prototype_loss', 0):.6f}")
        print(f"Z stats:")
        print(f"  std_ema: {logs.get('z_std_ema', 0):.6f}")
        print(f"  grad_norm: {logs.get('grad_norm', 0):.6f}")

        # 警告チェック
        prototype_loss = logs.get("prototype_loss", 0)
        recon = logs.get("recon", 0)

        if epoch > 30:
            if prototype_loss > 2.0:
                print(f"\n⚠️  Prototype lossが高すぎます ({prototype_loss:.3f})")
            elif prototype_loss < 0.001:
                print(f"\n⚠️  Prototype lossが低すぎます ({prototype_loss:.6f})")
            elif 0.1 <= prototype_loss <= 1.0:
                print(
                    f"\n✅ Prototype lossが良好な範囲です ({prototype_loss:.3f})"
                )

        print(f"{'='*70}\n")


def train_model(resume_checkpoint=True, initial_epoch_override=None):
    """
    モデルを訓練

    Args:
        resume_checkpoint: チェックポイントから再開するか
        initial_epoch_override: 開始エポックを手動指定（Noneの場合は自動検出）
    """
    print("=" * 70)
    print("LearnablePrototypesモデル訓練開始")
    print("=" * 70)

    # データセット準備
    print("\n[1] データセット読み込み中...")
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=16)
    dataset = dataset.repeat()  # 重要: repeatを追加
    steps_per_epoch = 87
    print(f"✓ データセット準備完了 (steps_per_epoch={steps_per_epoch})")

    # モデル構築
    print("\n[2] モデル構築中...")
    model = TimeWiseCVAE(steps_per_epoch=steps_per_epoch)

    # ★重要: モデルを正しくビルド
    print("  モデルをビルド中...")
    dummy_x = tf.zeros((1, TIME_LENGTH, 1), dtype=tf.float32)
    dummy_cond = tf.zeros((1, 4), dtype=tf.float32)

    # callメソッドでビルド
    _ = model((dummy_x, dummy_cond), training=False)

    print("✓ モデル構築完了")

    # モデル構造を確認
    print("\n[3] モデル構造確認...")
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    encoder_params = sum(
        [tf.size(v).numpy() for v in model.encoder.trainable_variables]
    )
    decoder_params = sum(
        [tf.size(v).numpy() for v in model.decoder.trainable_variables]
    )
    prototype_params = sum(
        [tf.size(v).numpy() for v in model.prototypes.trainable_variables]
    )

    print(f"  総パラメータ数: {total_params:,}")
    print(f"  - Encoder: {encoder_params:,}")
    print(f"  - Decoder: {decoder_params:,}")
    print(f"  - Prototypes: {prototype_params:,}")
    print(f"  Prototypes shape: {model.prototypes.prototypes.shape}")

    # オプティマイザ設定
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer)

    # チェックポイント管理
    checkpoint_path = "checkpoints/best_model.weights.h5"
    initial_epoch = 0

    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"\n[4] チェックポイント検出: {checkpoint_path}")

        try:
            # 重みをロード
            model.load_weights(checkpoint_path)
            print("✓ チェックポイントから重みをロードしました")

            # 開始エポックを決定
            if initial_epoch_override is not None:
                initial_epoch = initial_epoch_override
                print(f"  手動指定: 初期エポック = {initial_epoch}")
            elif os.path.exists("training_log.csv"):
                with open("training_log.csv", "r") as f:
                    reader = list(csv.reader(f))
                    if len(reader) > 1:
                        last_epoch = int(reader[-1][0])
                        initial_epoch = last_epoch + 1
                        print(f"  CSVログより: 初期エポック = {initial_epoch}")

            print(f"→ エポック {initial_epoch} から学習を再開します")

        except Exception as e:
            print(f"⚠️  チェックポイントのロードに失敗: {e}")
            print("→ 新規学習を開始します")
            initial_epoch = 0
    else:
        print("\n[4] 新規学習を開始します")

    # ディレクトリ作成
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # コールバック設定
    print("\n[5] コールバック設定...")
    callbacks = [
        # エポックごとのチェックポイント
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/epoch_{epoch:03d}.weights.h5",
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        ),
        # ベストモデル保存
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        ),
        # CSVログ
        CSVLogger("training_log.csv"),
        # 詳細ログ
        DetailedProgressLogger(),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=False,
        ),
        # 学習率スケジューリング
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("✓ コールバック設定完了")

    # 訓練開始
    print("\n[6] 訓練開始")
    print("=" * 70)
    print("監視すべき指標:")
    print("  1. prototype_loss: 0.1-1.0 の範囲（目標）")
    print("  2. recon: 0.005以下まで減少（目標）")
    print("  3. z_std_ema: 0.5-2.0の範囲（目標）")
    print("=" * 70)

    total_epochs = 200

    try:
        history = model.fit(
            dataset,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        print("\n" + "=" * 70)
        print("訓練完了！")
        print("=" * 70)

        # 最終結果表示
        if history.history:
            final_loss = history.history["loss"][-1]
            final_prototype = history.history.get("prototype_loss", [0])[-1]
            final_recon = history.history.get("recon", [0])[-1]

            print(f"\n最終結果:")
            print(f"  Loss: {final_loss:.6f}")
            print(f"  Prototype Loss: {final_prototype:.6f}")
            print(f"  Reconstruction: {final_recon:.6f}")

            # 評価
            if 0.1 <= final_prototype <= 1.0:
                print("\n✅ Prototype lossが良好な範囲です")
            else:
                print(f"\n⚠️  Prototype lossが範囲外: {final_prototype:.6f}")

            if final_recon < 0.005:
                print("✅ 再構成誤差が十分小さいです")
            else:
                print(f"⚠️  再構成誤差が大きい: {final_recon:.6f}")

    except KeyboardInterrupt:
        print("\n訓練が中断されました")
        print("最後のチェックポイントが保存されています")

    print("\n保存場所:")
    print(f"  {checkpoint_path} - ベストモデル")
    print("  checkpoints/epoch_XXX.weights.h5 - 各エポック")
    print("  training_log.csv - 訓練ログ")
    print("  logs/ - TensorBoard")

    print("\n次のステップ:")
    print("  1. TensorBoardで確認:")
    print("     $ tensorboard --logdir=logs")
    print("  2. 推論テスト:")
    print("     $ python inference.py")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LearnablePrototypesモデルの訓練"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="チェックポイントから再開しない（新規訓練）",
    )
    parser.add_argument(
        "--initial-epoch", type=int, default=None, help="開始エポックを手動指定"
    )

    args = parser.parse_args()

    print("Starting training...")
    train_model(
        resume_checkpoint=not args.no_resume,
        initial_epoch_override=args.initial_epoch,
    )
    print("Training completed.")
