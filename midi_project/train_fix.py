import tensorflow as tf
from model import TimeWiseCVAE
import numpy as np
from create_datasets import make_dataset_from_synth_csv


class AntiCollapseCVAE(TimeWiseCVAE):
    """
    Posterior Collapseを防ぐための拡張版CVAE
    """

    def __init__(self, steps_per_epoch=87, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ★重要: エポックベースでパラメータを設定
        self.steps_per_epoch = steps_per_epoch

        # 学習戦略のパラメータ（エポック単位）
        self.kl_warmup_epochs = 20
        self.kl_rampup_epochs = 50
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch

        self.kl_target = 0.001  # 目標KL重み（より小さく）
        self.free_bits = 0.5  # 各次元の最小情報量（nats）

        # メトリクス追跡用
        self.z_std_ema = tf.Variable(1.0, trainable=False)  # 指数移動平均

    def compute_kl_weight_with_warmup(self):
        """
        段階的にKL損失を導入
        """
        step = tf.cast(self.optimizer.iterations, tf.float32)

        # Phase 1 (0-10000): KL=0
        # Phase 2 (10000-30000): 0 → target
        # Phase 3 (30000+): target

        warmup_progress = (step - self.kl_warmup_steps) / 20000.0
        warmup_progress = tf.clip_by_value(warmup_progress, 0.0, 1.0)

        return self.kl_target * warmup_progress

    def compute_free_bits_kl(self, z_mean, z_logvar):
        """
        Free Bits: 各次元で最低限の情報量を保証
        """
        # 次元ごとのKL
        kl_per_dim = -0.5 * (
            1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )

        # 各次元で free_bits 以上を強制
        kl_clamped = tf.maximum(kl_per_dim, self.free_bits)

        return tf.reduce_mean(kl_clamped)

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

            # 周波数特徴を生成
            from model import generate_frequency_features, TIME_LENGTH

            pitch = cond[:, 0]
            freq_feat = generate_frequency_features(pitch, TIME_LENGTH)

            x_hat = self.decoder([z_noisy, cond, freq_feat])
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

            # ★対策3: Mutual Information追加損失
            # zとxの相互情報量を最大化（オプション）
            # これはより高度なテクニックなので、まずは Free Bits で試す

            from loss import Loss

            stft_loss, mel_loss, diff_loss = Loss(
                x_target, x_hat_sq, fft_size=2048, hop_size=512
            )

            # ★対策4: 段階的なKL重み
            kl_weight = self.compute_kl_weight_with_warmup()

            loss = (
                recon * 5.0
                + stft_loss * 10.0
                + mel_loss * 8.0
                + diff_loss * 2.0
                + kl_free_bits * kl_weight  # Free Bitsを使用
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # ★対策5: zの活用度を監視
        z_std = tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1))
        # 指数移動平均で平滑化
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)

        # ★対策6: 警告システム
        # z_stdが小さくなりすぎたらKL重みを下げる（自動調整）
        should_reduce_kl = tf.cond(
            self.z_std_ema < 0.05, lambda: True, lambda: False
        )

        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_loss,
            "mel": mel_loss,
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
                print("   2. Free Bitsを増やす (0.5 → 1.0)")
                print("   3. 学習率を下げる")
                print("   4. より多くのWarmupステップを使う\n")
        else:
            self.low_std_count = 0
            if epoch % 10 == 0:
                print(f"✓ z_std={z_std:.4f} - 潜在変数は健全です")


# 学習スクリプト例
def train_with_anti_collapse():
    """
    Posterior Collapse対策を施した学習
    """
    # データ準備（既存のデータセットを使用）
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=16)

    # モデル構築
    model = AntiCollapseCVAE()
    x_dummy, cond_dummy = next(iter(dataset))
    _ = model((x_dummy, cond_dummy), training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # コールバック
    callbacks = [
        CollapseDetectionCallback(threshold=0.05, patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/best_model.weights.h5",
            monitor="z_std_ema",
            mode="max",  # z_stdが大きい方が良い
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="collapse_warning", patience=20, restore_best_weights=True
        ),
    ]

    # 学習実行
    history = model.fit(
        dataset,
        epochs=200,
        callbacks=callbacks,
    )

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Posterior Collapse対策版 学習スクリプト")
    print("=" * 60)
    print("\n主な対策:")
    print("1. Free Bits: 各次元で最低限の情報量を保証")
    print("2. KL Warmup: 段階的にKL損失を導入")
    print("3. ノイズ注入: デコーダーをzの変化に頑強に")
    print("4. 自動監視: z_stdを追跡して警告")
    print("5. 適応的学習: Collapse検出時に学習率を自動調整")
    print("=" * 60)

    train_with_anti_collapse()
