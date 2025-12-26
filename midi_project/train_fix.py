import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
from model import TimeWiseCVAE, TIME_LENGTH, LATENT_STEPS, LATENT_DIM
from create_datasets import make_dataset_from_synth_csv


class TrainingState:
    """
    訓練状態を管理するクラス
    """

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, "training_state.json")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_state(self, epoch, step, best_loss, history):
        """
        訓練状態を保存
        """
        state = {
            "epoch": int(epoch),
            "step": int(step),
            "best_loss": float(best_loss),
            "history": {
                k: [float(v) for v in vals] for k, vals in history.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

        print(f"  ✓ 訓練状態を保存: {self.state_file}")

    def load_state(self):
        """
        訓練状態を読み込み
        """
        if not os.path.exists(self.state_file):
            return None

        with open(self.state_file, "r") as f:
            state = json.load(f)

        print(f"  ✓ 訓練状態を読み込み: {self.state_file}")
        print(f"    前回のエポック: {state['epoch']}")
        print(f"    前回のステップ: {state['step']}")
        print(f"    ベスト損失: {state['best_loss']:.6f}")

        return state

    def get_latest_checkpoint(self):
        """
        最新のチェックポイントを取得
        """
        checkpoints = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("epoch_") and f.endswith(".weights.h5")
        ]

        if not checkpoints:
            return None

        # エポック番号でソート
        checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        latest = os.path.join(self.checkpoint_dir, checkpoints[-1])

        print(f"  ✓ 最新チェックポイント: {latest}")
        return latest


class ProgressCallback(tf.keras.callbacks.Callback):
    """
    学習進捗を詳細に表示するコールバック
    """

    def __init__(self, steps_per_epoch):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} 完了")
        print(f"{'='*60}")
        print(f"  loss:       {logs.get('loss', 0):.6f}")
        print(f"  recon:      {logs.get('recon', 0):.6f}")
        print(f"  stft:       {logs.get('stft', 0):.6f}")
        print(f"  mel:        {logs.get('mel', 0):.6f}")
        print(f"  kl:         {logs.get('kl', 0):.6f}")
        print(f"  kl_weight:  {logs.get('kl_weight', 0):.6f}")
        print(f"  z_std_ema:  {logs.get('z_std_ema', 0):.6f}")
        print(f"  grad_norm:  {logs.get('grad_norm', 0):.6f}")

        # 警告チェック
        if logs.get("z_std_ema", 1.0) < 0.05:
            print(f"\n⚠️  WARNING: Posterior Collapse の兆候")


class GenerationTestCallback(tf.keras.callbacks.Callback):
    """
    定期的に音声を生成してテストするコールバック
    """

    def __init__(self, test_interval=10, output_dir="generation_tests"):
        super().__init__()
        self.test_interval = test_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.test_interval != 0:
            return

        print(f"\n[生成テスト] Epoch {epoch + 1}")

        import soundfile as sf

        # テスト条件
        test_cases = [
            (60, (1, 0, 0), "screech"),
            (60, (0, 1, 0), "acid"),
            (60, (0, 0, 1), "pluck"),
        ]

        for pitch, cond, name in test_cases:
            pitch_norm = (pitch - 36.0) / 35.0
            cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

            # ランダムな潜在変数
            z = tf.random.normal((1, LATENT_STEPS, LATENT_DIM), stddev=0.7)

            # 生成
            x_hat = self.model.decoder([z, cond_vector])
            x_hat = tf.squeeze(x_hat).numpy()

            # 正規化
            max_val = np.max(np.abs(x_hat))
            if max_val > 1e-6:
                x_hat = x_hat / max_val * 0.95

            # 保存
            filename = os.path.join(
                self.output_dir, f"epoch_{epoch+1:03d}_{name}.wav"
            )
            sf.write(filename, x_hat, samplerate=48000)

        print(
            f"  ✓ テスト音声を生成: {self.output_dir}/epoch_{epoch+1:03d}_*.wav"
        )


class CollapseDetectionCallback(tf.keras.callbacks.Callback):
    """
    Posterior Collapse を検出して警告
    """

    def __init__(self, threshold=0.05, patience=5):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.low_std_count = 0

    def on_epoch_end(self, epoch, logs=None):
        z_std = logs.get("z_std_ema", 1.0)

        if z_std < self.threshold:
            self.low_std_count += 1
            print(f"\n⚠️  WARNING: z_std_ema={z_std:.4f} < {self.threshold}")
            print(
                f"   Posterior Collapse の兆候 ({self.low_std_count}/{self.patience})"
            )

            if self.low_std_count >= self.patience:
                print("\n🚨 CRITICAL: Posterior Collapse 検出！")
                print("   推奨対策:")
                print("   1. KL重みを1/10に減らす")
                print("   2. Free Bitsを増やす (0.8 → 1.2)")
                print("   3. 学習率を下げる")
        else:
            self.low_std_count = 0


def train_model(
    dataset_path="dataset.csv",
    batch_size=16,
    epochs=200,
    initial_epoch=0,
    checkpoint_dir="checkpoints",
    resume=True,
    save_interval=5,
):
    """
    モデルを訓練

    Args:
        dataset_path: データセットのCSVパス
        batch_size: バッチサイズ
        epochs: 総エポック数
        initial_epoch: 開始エポック（通常は0、再開時は自動設定）
        checkpoint_dir: チェックポイント保存ディレクトリ
        resume: True の場合、前回の学習から再開
        save_interval: 何エポックごとに保存するか
    """
    print("=" * 60)
    print("表現力豊かな音色モデル 訓練開始")
    print("=" * 60)

    # 訓練状態管理
    training_state = TrainingState(checkpoint_dir)

    # データセット準備
    print("\n[1] データセット準備中...")
    dataset = make_dataset_from_synth_csv(dataset_path, batch_size=batch_size)

    # データセットサイズを取得
    import pandas as pd

    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    steps_per_epoch = total_samples // batch_size

    print(f"  総サンプル数: {total_samples}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  ステップ/エポック: {steps_per_epoch}")

    dataset = dataset.repeat()

    # モデル構築
    print("\n[2] モデル構築中...")
    model = TimeWiseCVAE(steps_per_epoch=steps_per_epoch)

    # ダミーデータでビルド
    x_dummy, cond_dummy = next(iter(dataset))
    _ = model((x_dummy, cond_dummy), training=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    print(f"  エンコーダー: {model.encoder.count_params():,} パラメータ")
    print(f"  デコーダー: {model.decoder.count_params():,} パラメータ")
    print(f"  合計: {model.count_params():,} パラメータ")

    # 学習の再開処理
    best_loss = float("inf")
    history = {
        "loss": [],
        "recon": [],
        "stft": [],
        "mel": [],
        "kl": [],
        "kl_weight": [],
        "z_std_ema": [],
        "grad_norm": [],
    }

    if resume:
        print("\n[3] 前回の学習状態を確認中...")
        state = training_state.load_state()

        if state is not None:
            initial_epoch = state["epoch"]
            best_loss = state["best_loss"]
            history = state["history"]

            # 最新のチェックポイントを読み込み
            latest_checkpoint = training_state.get_latest_checkpoint()
            if latest_checkpoint:
                model.load_weights(latest_checkpoint)
                print(f"  ✓ 重みを読み込みました")
                print(f"\n  → Epoch {initial_epoch + 1} から再開します")
            else:
                print(f"  ⚠️  チェックポイントが見つかりません")
                print(f"  → Epoch 1 から新規に開始します")
                initial_epoch = 0
        else:
            print(f"  訓練状態ファイルが見つかりません")
            print(f"  → 新規に訓練を開始します")
    else:
        print("\n[3] 新規訓練を開始します")

    # コールバック設定
    print("\n[4] コールバック設定中...")
    callbacks = [
        # 進捗表示
        ProgressCallback(steps_per_epoch),
        # Collapse検出
        CollapseDetectionCallback(threshold=0.05, patience=5),
        # 定期的な生成テスト
        GenerationTestCallback(test_interval=10),
        # 定期的なチェックポイント保存
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: (
                (
                    model.save_weights(
                        os.path.join(
                            checkpoint_dir, f"epoch_{epoch+1:03d}.weights.h5"
                        )
                    )
                    if (epoch + 1) % save_interval == 0
                    else None
                ),
                (
                    training_state.save_state(
                        epoch + 1,
                        (epoch + 1) * steps_per_epoch,
                        logs.get("loss", float("inf")),
                        {
                            k: history[k] + [logs.get(k, 0)]
                            for k in history.keys()
                        },
                    )
                    if (epoch + 1) % save_interval == 0
                    else None
                ),
                (
                    print(f"\n  ✓ Epoch {epoch+1} を保存しました")
                    if (epoch + 1) % save_interval == 0
                    else None
                ),
            )
        ),
        # ベストモデル保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_model.weights.h5"),
            monitor="loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        # 学習率削減
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
        ),
        # CSVログ
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, "training_log.csv"), append=True
        ),
    ]

    # 学習戦略の表示
    print("\n[5] 学習戦略:")
    print(f"  開始エポック: {initial_epoch + 1}")
    print(f"  終了エポック: {epochs}")
    print(f"  KL Warmup: {model.kl_warmup_epochs} エポック")
    print(f"  KL Rampup: {model.kl_rampup_epochs} エポック")
    print(f"  KL Target: {model.kl_target}")
    print(f"  Free Bits: {model.free_bits}")
    print(f"  保存間隔: {save_interval} エポックごと")

    # 学習開始
    print("\n[6] 学習開始...")
    print("=" * 60)

    try:
        history_obj = model.fit(
            dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )

        # 最終状態を保存
        final_logs = history_obj.history
        for k in history.keys():
            if k in final_logs:
                history[k].extend(final_logs[k])

        training_state.save_state(
            epochs,
            epochs * steps_per_epoch,
            min(final_logs.get("loss", [float("inf")])),
            history,
        )

        print("\n" + "=" * 60)
        print("訓練完了！")
        print("=" * 60)
        print(f"  最終 loss: {final_logs['loss'][-1]:.6f}")
        print(f"  最終 z_std_ema: {final_logs['z_std_ema'][-1]:.6f}")
        print(f"  最終 kl_weight: {final_logs['kl_weight'][-1]:.6f}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("訓練が中断されました")
        print("=" * 60)
        print("次回実行時に resume=True で再開できます")

        # 中断時の状態を保存
        current_epoch = model.optimizer.iterations.numpy() // steps_per_epoch
        training_state.save_state(
            current_epoch,
            model.optimizer.iterations.numpy(),
            best_loss,
            history,
        )

        # 中断時のモデルを保存
        interrupt_path = os.path.join(checkpoint_dir, "interrupted.weights.h5")
        model.save_weights(interrupt_path)
        print(f"  ✓ 中断時の重みを保存: {interrupt_path}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="表現力豊かな音色モデルの訓練")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.csv",
        help="データセットのCSVパス",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="バッチサイズ"
    )
    parser.add_argument("--epochs", type=int, default=200, help="総エポック数")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="チェックポイント保存ディレクトリ",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="前回の学習を無視して新規に開始",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="何エポックごとに保存するか",
    )

    args = parser.parse_args()

    print("\n設定:")
    print(f"  データセット: {args.dataset}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  総エポック数: {args.epochs}")
    print(f"  チェックポイントDir: {args.checkpoint_dir}")
    print(f"  再開: {not args.no_resume}")
    print(f"  保存間隔: {args.save_interval} エポック")

    model, history = train_model(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        save_interval=args.save_interval,
    )
