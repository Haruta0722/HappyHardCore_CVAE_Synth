import tensorflow as tf
import numpy as np
import os
from model import TimeWiseCVAE, TIME_LENGTH
from create_datasets import make_dataset_from_synth_csv

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
            monitor="loss",
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1,
        ),
        # Early Stopping（オプション）
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=50,
            restore_best_weights=True,
            verbose=1,
        ),
        # カスタムコールバック：学習状況の詳細表示
        DetailedLogger(),
        # ★新規: KL損失の監視コールバック
        KLMonitor(),
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


class KLMonitor(tf.keras.callbacks.Callback):
    """
    ★新規: KL損失の監視と警告
    z=random問題を検出するためのコールバック
    """

    def __init__(self):
        super().__init__()
        self.kl_history = []
        self.z_std_history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        kl_loss = logs.get("kl", 0)
        z_std = logs.get("z_std_ema", 0)
        kl_weight = logs.get("kl_weight", 0)

        self.kl_history.append(kl_loss)
        self.z_std_history.append(z_std)

        # 警告チェック
        warnings = []

        # KL損失が低すぎる（posterior collapse）
        if epoch > 40 and kl_loss < 0.1:
            warnings.append(
                "⚠️  KL損失が低すぎます。Posterior collapseの可能性があります。"
            )

        # zの標準偏差が小さすぎる
        if epoch > 40 and z_std < 0.3:
            warnings.append(
                "⚠️  Zの標準偏差が小さすぎます。潜在空間が使われていない可能性があります。"
            )

        # zの標準偏差が大きすぎる
        if z_std > 3.0:
            warnings.append("⚠️  Zの標準偏差が大きすぎます。学習が不安定です。")

        # KL weightが適切に増加しているか
        if epoch == 30 and kl_weight < 0.00001:
            warnings.append("⚠️  KL weightの増加が遅すぎます。")

        # 警告を表示
        if warnings:
            print("\n" + "🔍 診断メッセージ ".center(60, "="))
            for warning in warnings:
                print(warning)
            print("=" * 60 + "\n")

        # 良好な状態を報告
        if (
            epoch > 60
            and 0.5 < kl_loss < 5.0
            and 0.5 < z_std < 2.0
            and kl_weight > 0.0001
        ):
            print("\n✅ 潜在変数の学習が良好です！")


class SynthesisTest(tf.keras.callbacks.Callback):
    """
    ★新規: 定期的に合成テストを実行
    z=0とz=randomでの音声合成をテスト
    """

    def __init__(self, test_interval=10, output_dir="test_outputs"):
        super().__init__()
        self.test_interval = test_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.test_interval != 0:
            return

        print(f"\n🎵 合成テスト (Epoch {epoch+1}) ".center(60, "="))

        # テスト用条件: [pitch, screech, acid, pluck]
        test_conditions = [
            ([0.5, 0.0, 0.0, 1.0], "pluck"),
            ([0.3, 0.0, 1.0, 0.0], "acid"),
            ([0.7, 1.0, 0.0, 0.0], "screech"),
        ]

        for cond_values, timbre_name in test_conditions:
            cond = tf.constant([cond_values], dtype=tf.float32)

            # z=0でテスト
            z_zero = tf.zeros((1, self.model.decoder.input[0].shape[1], 64))
            try:
                output_zero = self.model.decoder([z_zero, cond], training=False)
                rms_zero = tf.sqrt(tf.reduce_mean(tf.square(output_zero)))
                status_zero = "✓" if rms_zero > 0.01 else "✗"
                print(
                    f"  {timbre_name} (z=0): RMS={rms_zero:.4f} {status_zero}"
                )
            except Exception as e:
                print(f"  {timbre_name} (z=0): エラー - {str(e)}")

            # z=randomでテスト
            z_random = tf.random.normal(
                (1, self.model.decoder.input[0].shape[1], 64)
            )
            try:
                output_random = self.model.decoder(
                    [z_random, cond], training=False
                )
                rms_random = tf.sqrt(tf.reduce_mean(tf.square(output_random)))
                status_random = "✓" if rms_random > 0.01 else "✗"
                print(
                    f"  {timbre_name} (z=random): RMS={rms_random:.4f} {status_random}"
                )
            except Exception as e:
                print(f"  {timbre_name} (z=random): エラー - {str(e)}")

        print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("改善版 DDSP風モデル 訓練スクリプト")
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
    )
    train_dataset = train_dataset.repeat()

    # 1エポックあたりのステップ数を計算
    steps_per_epoch = 87  # あなたのデータセットサイズに合わせて変更

    print(f"✓ データセット読み込み完了")
    print(f"  Steps per epoch: {steps_per_epoch}")

    # モデル構築
    print("\n[2] モデル構築中...")
    model = TimeWiseCVAE(steps_per_epoch=steps_per_epoch)

    # オプティマイザ
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,  # 勾配クリッピング
    )
    model.compile(optimizer=optimizer)

    # ダミーデータでモデルをビルド
    dummy_x = tf.zeros((1, TIME_LENGTH, 1))
    dummy_cond = tf.zeros((1, 4))
    _ = model((dummy_x, dummy_cond), training=False)

    print("✓ モデル構築完了")
    print("\n[Encoder]")
    model.encoder.summary()
    print("\n[Decoder]")
    model.decoder.summary()

    # パラメータ数を表示
    total_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"\n総パラメータ数: {total_params:,}")

    # ★改善点の確認
    print("\n" + "=" * 60)
    print("改善ポイント:")
    print("=" * 60)
    print("✓ KL warmup: 30エポック（従来: 20）")
    print("✓ KL rampup: 60エポック（従来: 50）")
    print("✓ KL target: 0.0003（従来: 0.0005）")
    print("✓ Free bits: 1.0（従来: 0.8）")
    print("✓ z_logvar初期値: -2.0（従来: -3.0）")
    print("✓ 音色特性をcondから直接生成")
    print("✓ screechノイズ比: 0.3（従来: 0.6）")
    print("=" * 60)

    # コールバック設定
    print("\n[3] コールバック設定...")
    callbacks = create_callbacks()
    # ★新規: 合成テストコールバックを追加
    callbacks.append(SynthesisTest(test_interval=10))

    print("✓ コールバック設定完了")
    print("  - ModelCheckpoint (10エポックごと)")
    print("  - Best model checkpoint")
    print("  - TensorBoard")
    print("  - ReduceLROnPlateau")
    print("  - EarlyStopping")
    print("  - DetailedLogger")
    print("  - KLMonitor (新規)")
    print("  - SynthesisTest (新規)")

    # 訓練開始
    print("\n[4] 訓練開始")
    print("=" * 60)
    print("学習中の注目ポイント:")
    print("  1. z_std_ema が 0.5-2.0 の範囲に収束するか")
    print("  2. KL損失が徐々に増加するか（0.5-5.0が目標）")
    print("  3. 30エポック以降でKL weightが増加し始めるか")
    print("  4. 合成テストでz=0とz=randomの両方で音が出るか")
    print("=" * 60 + "\n")

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
        final_z_std = history.history["z_std_ema"][-1]

        print(f"\n最終結果:")
        print(f"  Loss: {final_loss:.6f}")
        print(f"  Reconstruction: {final_recon:.6f}")
        print(f"  KL: {final_kl:.6f}")
        print(f"  Z std EMA: {final_z_std:.6f}")

        # 最良エポックの情報
        best_epoch = np.argmin(history.history["loss"]) + 1
        best_loss = np.min(history.history["loss"])
        print(f"\n最良エポック: {best_epoch}")
        print(f"  Loss: {best_loss:.6f}")

        # 学習の健全性チェック
        print("\n" + "=" * 60)
        print("学習の健全性チェック:")
        print("=" * 60)

        checks = []
        if 0.5 <= final_z_std <= 2.0:
            checks.append("✓ Z標準偏差が適切な範囲です")
        else:
            checks.append(f"✗ Z標準偏差が範囲外です ({final_z_std:.2f})")

        if 0.5 <= final_kl <= 10.0:
            checks.append("✓ KL損失が適切な範囲です")
        else:
            checks.append(f"✗ KL損失が範囲外です ({final_kl:.2f})")

        if final_recon < 0.01:
            checks.append("✓ 再構成誤差が十分小さいです")
        else:
            checks.append(f"⚠️  再構成誤差が大きいです ({final_recon:.4f})")

        for check in checks:
            print(check)

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n訓練が中断されました")
        print("最後のチェックポイントが保存されています")

    print("\n保存場所:")
    print("  weights/best_model.weights.h5 - 最良モデル")
    print("  weights/epoch_XXX.weights.h5 - 各エポックのチェックポイント")
    print("  logs/ - TensorBoardログ")
    print("  test_outputs/ - 合成テスト出力")

    print("\n次のステップ:")
    print("  1. TensorBoardで訓練曲線を確認")
    print("     $ tensorboard --logdir=logs")
    print("  2. 以下の指標を確認:")
    print("     - z_std_ema: 0.5-2.0の範囲にあるか")
    print("     - kl: 0.5-5.0の範囲で推移しているか")
    print("     - kl_weight: 60エポック以降で0.0003に達しているか")
    print("  3. inference_improved.py で推論テスト")
    print("     - z=0でpluckの急速減衰を確認")
    print("     - z=0でacidのうねりを確認")
    print("     - z=0でscreechのノイズ量を確認")
    print("     - z=randomで音が正常に生成されるか確認")
    print("=" * 60)


if __name__ == "__main__":
    main()
