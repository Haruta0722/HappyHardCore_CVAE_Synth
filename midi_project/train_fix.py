import tensorflow as tf
from model import (
    TimeWiseCVAE,
    TIME_LENGTH,
    LATENT_STEPS,
    LATENT_DIM,
    recon_weight,
    STFT_weight,
    mel_weight,
    diff_weight,
)  # 強力な条件付けモデルをインポート
import numpy as np
from create_datasets import make_dataset_from_synth_csv


class AntiCollapseStrongCVAE(TimeWiseCVAE):
    """
    強力な条件付け + Posterior Collapse対策版CVAE
    freq_featは使用しない
    """

    def __init__(self, steps_per_epoch=87, *args, **kwargs):
        super().__init__(steps_per_epoch=steps_per_epoch, *args, **kwargs)
        # 親クラスで既に設定済み

    def add_gaussian_noise_to_z(self, z, noise_scale=0.1):
        """
        学習中にzにノイズを追加して、デコーダーを頑強にする
        """
        noise = tf.random.normal(tf.shape(z), stddev=noise_scale)
        return z + noise

    def train_step(self, data):
        x, cond = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder([x, cond])
            z = self.sample_z(z_mean, z_logvar)

            # ★対策1: 学習中にzにノイズを追加
            # デコーダーがzの小さな変化にも対応できるようにする
            z_noisy = self.add_gaussian_noise_to_z(z, noise_scale=0.05)

            # ★重要: freq_featは使わない！
            # 強力な条件付けモデルは条件ベクトルとzだけで生成
            x_hat = self.decoder([z_noisy, cond])

            x_hat = x_hat[:, :TIME_LENGTH, :]

            x_target = tf.squeeze(x, axis=-1)
            x_hat_sq = tf.squeeze(x_hat, axis=-1)

            # 損失計算
            recon = tf.reduce_mean(tf.square(x_target - x_hat_sq))

            # ★対策2: Free Bits KL
            kl_free_bits = self.compute_free_bits_kl(z_mean, z_logvar)

            # 通常のKLも計算（監視用）
            kl_standard = -0.5 * tf.reduce_mean(
                1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            )

            from loss import Loss

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★対策3: 段階的なKL重み
            kl_weight = self.compute_kl_weight()

            loss = (
                recon * recon_weight  # 5.0
                + stft_loss * STFT_weight  # 15.0
                + mel_loss * mel_weight  # 10.0
                + diff_loss * diff_weight  # 3.0
                + kl_free_bits * kl_weight
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # ★対策4: zの活用度を監視
        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))
        # 指数移動平均で平滑化
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)

        # ★対策5: 警告システム
        should_reduce_kl = tf.cond(
            self.z_std_ema < 0.05, lambda: True, lambda: False
        )

        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_loss,
            "mel": mel_loss,
            "diff": diff_loss,
            "kl_standard": kl_standard,
            "kl_free_bits": kl_free_bits,
            "kl_weight": kl_weight,
            "z_std": z_std,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
            "collapse_warning": tf.cast(should_reduce_kl, tf.float32),
        }

    def sample_z(self, z_mean, z_logvar):
        """
        Reparameterization trick
        """
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps


# カスタムコールバック: Collapse検出
class CollapseDetectionCallback(tf.keras.callbacks.Callback):
    """
    学習中にPosterior Collapseを検出して警告
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
            print(f"\n⚠️  WARNING: z_std={z_std:.4f} < {self.threshold}")
            print(
                f"   Posterior Collapse の兆候 ({self.low_std_count}/{self.patience})"
            )

            if self.low_std_count >= self.patience:
                print("\n🚨 CRITICAL: Posterior Collapse 検出！")
                print("   推奨対策:")
                print("   1. KL重みを1/10に減らす")
                print("   2. Free Bitsを増やす (0.8 → 1.2)")
                print("   3. 学習率を下げる")
                print("   4. より多くのWarmupステップを使う\n")
        else:
            self.low_std_count = 0
            if epoch % 10 == 0:
                print(f"✓ z_std={z_std:.4f} - 潜在変数は健全です")


class ConditionMonitorCallback(tf.keras.callbacks.Callback):
    """
    条件ベクトルの効果を監視
    定期的に異なる条件で生成して保存
    """

    def __init__(self, test_pitch=60, check_every=10):
        super().__init__()
        self.test_pitch = test_pitch
        self.check_every = check_every

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_every != 0:
            return

        print(f"\n[Epoch {epoch}] 条件別生成テスト...")

        import soundfile as sf

        pitch_norm = (self.test_pitch - 36.0) / 35.0
        conditions = {
            "screech": (1, 0, 0),
            "acid": (0, 1, 0),
            "pluck": (0, 0, 1),
        }

        for name, cond in conditions.items():
            cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

            # ランダムなzで生成
            z = tf.random.normal((1, LATENT_STEPS, LATENT_DIM), stddev=0.5)

            x_hat = self.model.decoder([z, cond_vector])
            x_hat = tf.squeeze(x_hat).numpy()

            # 正規化
            max_val = np.max(np.abs(x_hat))
            if max_val > 1e-6:
                x_hat = x_hat / max_val * 0.95

            filename = f"monitor/epoch_{epoch:03d}_{name}.wav"
            sf.write(filename, x_hat, samplerate=48000)

        print(f"  ✓ 保存: monitor/epoch_{epoch:03d}_*.wav")


# 学習スクリプト
def train_with_strong_conditioning(batch_size=16, epochs=200):
    """
    強力な条件付けモデルで学習
    """
    print("=" * 60)
    print("強力な条件付けモデル 学習開始")
    print("=" * 60)

    # データ準備
    print("\n[1] データセット読み込み中...")
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=batch_size)

    # データセットのステップ数を計算
    # dataset.csvの行数を確認してください
    # 例: 348行のデータ、batch_size=16 → steps_per_epoch = 348//16 = 21

    # ★重要: 実際のデータサイズに合わせて変更
    import pandas as pd

    df = pd.read_csv("dataset.csv")
    total_samples = len(df)
    steps_per_epoch = total_samples // batch_size

    print(f"  総サンプル数: {total_samples}")
    print(f"  バッチサイズ: {batch_size}")
    print(f"  ステップ/エポック: {steps_per_epoch}")

    dataset = dataset.repeat()

    # モデル構築
    print("\n[2] モデル構築中...")
    model = AntiCollapseStrongCVAE(steps_per_epoch=steps_per_epoch)

    # モデルの初期化（build）
    x_dummy, cond_dummy = next(iter(dataset))
    _ = model((x_dummy, cond_dummy), training=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    print(f"  エンコーダー: {model.encoder.count_params():,} パラメータ")
    print(f"  デコーダー: {model.decoder.count_params():,} パラメータ")
    print(f"  合計: {model.count_params():,} パラメータ")

    # 学習戦略の表示
    print("\n[3] 学習戦略:")
    print(
        f"  KL Warmup: {model.kl_warmup_epochs} エポック ({model.kl_warmup_steps} ステップ)"
    )
    print(
        f"  KL Rampup: {model.kl_rampup_epochs} エポック ({model.kl_rampup_steps} ステップ)"
    )
    print(f"  KL Target: {model.kl_target}")
    print(f"  Free Bits: {model.free_bits}")

    # monitorディレクトリ作成
    import os

    os.makedirs("monitor", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # コールバック
    callbacks = [
        CollapseDetectionCallback(threshold=0.05, patience=5),
        ConditionMonitorCallback(test_pitch=60, check_every=10),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/epoch_{epoch:03d}.weights.h5",
            save_freq="epoch",
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/best_model.weights.h5",
            monitor="z_std_ema",
            mode="max",  # z_stdが大きい方が良い
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv", append=True),
    ]

    # 学習実行
    print("\n[4] 学習開始...")
    print("=" * 60)

    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("学習完了！")
    print("=" * 60)

    # 最終評価
    print("\n[5] 最終評価:")
    final_logs = history.history
    print(f"  最終 loss: {final_logs['loss'][-1]:.4f}")
    print(f"  最終 z_std_ema: {final_logs['z_std_ema'][-1]:.4f}")
    print(f"  最終 kl_weight: {final_logs['kl_weight'][-1]:.6f}")

    if final_logs["z_std_ema"][-1] > 0.1:
        print("\n✓ SUCCESS: 潜在変数は健全に活用されています")
    else:
        print("\n⚠️  WARNING: 潜在変数の活用が不十分です")

    return model, history


if __name__ == "__main__":
    print("=" * 60)
    print("強力な条件付けモデル 訓練スクリプト")
    print("=" * 60)
    print("\n特徴:")
    print("  1. freq_feat を使わない（条件とzのみで生成）")
    print("  2. TimbreEmbedding で音色を独立した空間に")
    print("  3. StrongFiLM + Attention で条件を強力に反映")
    print("  4. Free Bits + KL Annealing でCollapse防止")
    print("  5. 定期的な条件別生成で学習を監視")
    print("=" * 60)

    # 学習パラメータ
    BATCH_SIZE = 16  # メモリに応じて調整
    EPOCHS = 200

    print(f"\nバッチサイズ: {BATCH_SIZE}")
    print(f"エポック数: {EPOCHS}")
    print("\n学習を開始しますか？ (y/n)")
    # response = input().lower()

    # if response == 'y':
    model, history = train_with_strong_conditioning(
        batch_size=BATCH_SIZE, epochs=EPOCHS
    )

    print("\n✓ 訓練完了")
    print("  重み: checkpoints/best_model.weights.h5")
    print("  ログ: training_log.csv")
    print("  監視音声: monitor/epoch_*_*.wav")
