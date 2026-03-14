"""
train.py  ―  TimeWiseCVAE 訓練スクリプト

使い方:
    python train.py                          # デフォルト設定で学習
    python train.py --epochs 300            # エポック数を指定
    python train.py --resume                # チェックポイントから再開
    python train.py --csv path/to/data.csv  # CSVを指定

出力ディレクトリ構造:
    checkpoints/
        best_weights.weights.h5     ← val_loss が改善するたびに保存
        latest.weights.h5           ← エポックごとに上書き保存
    logs/
        fit/                        ← TensorBoard ログ
    weights_final.weights.h5        ← 学習完了後の最終重みファイル
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import TIME_LENGTH, TIMBRE_VOCAB
from cvae import TimeWiseCVAE

# ============================================================
# ハイパーパラメータ (argparse でも上書き可能)
# ============================================================
DEFAULTS = dict(
    csv="dataset.csv",  # データセットCSVのパス
    wav_root=".",  # WAVファイルのルートディレクトリ
    epochs=300,
    batch_size=16,
    lr=2e-4,
    val_split=0.1,  # 検証データの割合
    seed=42,
    out_dir=".",  # 重みファイルの出力先
    ckpt_dir="checkpoints",
    log_dir="logs/fit",
    resume=False,  # チェックポイントから再開するか
)

# 音色ラベル → ID のマッピング
TIMBRE_MAP = {"screech": 0, "acid": 1, "pluck": 2}

SR = 48000


# ============================================================
# データローダー
# ============================================================
def load_wav(path: str, target_length: int = TIME_LENGTH) -> np.ndarray:
    """
    WAVファイルを読み込み、モノラル・float32・固定長に整形して返す。

    - ステレオ → モノラル変換 (チャンネル平均)
    - 長さが足りない場合はゼロパディング
    - 長さが超過する場合は先頭からトリミング
    - ピーク正規化 (最大絶対値が 1.0 を超えないように)
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # モノラル化

    if sr != SR:
        # 簡易リサンプリング (scipy がない環境向けに線形補間で近似)
        orig_len = len(audio)
        target_sr_len = int(orig_len * SR / sr)
        audio = np.interp(
            np.linspace(0, orig_len - 1, target_sr_len),
            np.arange(orig_len),
            audio,
        ).astype(np.float32)

    # 長さ調整
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # ピーク正規化
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    return audio.astype(np.float32)  # [TIME_LENGTH]


def parse_csv(csv_path: str, wav_root: str):
    """
    CSVを読み込み、サンプルのリストを返す。

    各サンプル:
        {
            "path"     : str   (WAVファイルの絶対パス),
            "pitch"    : float (正規化ピッチ: (MIDI - 36) / 35),
            "timbre_id": int   (0=Screech, 1=Acid, 2=Pluck),
        }
    """
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = os.path.join(wav_root, row["path"].replace("\\", "/"))

            # 音高を正規化  (MIDI 36〜71 → 0.0〜1.0)
            midi = int(row["pitch"])
            pitch = (midi - 36) / 35.0

            # one-hot → 整数ID
            one_hot = [int(row["screech"]), int(row["acid"]), int(row["pluck"])]
            timbre_id = int(np.argmax(one_hot))

            samples.append(
                {
                    "path": wav_path,
                    "pitch": np.float32(pitch),
                    "timbre_id": np.int32(timbre_id),
                }
            )

    return samples


class AudioDataset:
    """
    WAVファイルリストを tf.data.Dataset に変換するクラス。

    __init__ 時に全WAVをメモリにキャッシュするオプションあり
    (データセット全体が ~1.2GB 以内ならキャッシュ推奨)。
    """

    def __init__(
        self,
        samples: list,
        batch_size: int,
        shuffle: bool = True,
        cache: bool = True,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(samples)

        # WAVをメモリにプリロード
        print(f"  WAVをメモリにロード中 ({self.n} ファイル)...")
        t0 = time.time()
        if cache:
            self._audio = np.zeros((self.n, TIME_LENGTH), dtype=np.float32)
            self._pitch = np.zeros(self.n, dtype=np.float32)
            self._timbre_id = np.zeros(self.n, dtype=np.int32)
            for i, s in enumerate(samples):
                self._audio[i] = load_wav(s["path"])
                self._pitch[i] = s["pitch"]
                self._timbre_id[i] = s["timbre_id"]
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{self.n}")
            self._cached = True
        else:
            self._cached = False
        print(f"  ロード完了 ({time.time()-t0:.1f}s)")

    def build(self) -> tf.data.Dataset:
        if self._cached:
            # メモリからTensorSlicesデータセットを作成
            audio_t = tf.constant(
                self._audio[:, :, None]
            )  # [N, TIME_LENGTH, 1]
            pitch_t = tf.constant(self._pitch)  # [N]
            tid_t = tf.constant(self._timbre_id)  # [N]
            ds = tf.data.Dataset.from_tensor_slices((audio_t, pitch_t, tid_t))
        else:
            # ファイルパスから都度ロード (メモリ節約モード)
            paths = [s["path"] for s in self.samples]
            pitches = [s["pitch"] for s in self.samples]
            tids = [s["timbre_id"] for s in self.samples]
            ds = tf.data.Dataset.from_tensor_slices((paths, pitches, tids))
            ds = ds.map(
                lambda p, pitch, tid: (
                    tf.py_function(
                        lambda x: load_wav(x.numpy().decode())[:, None],
                        [p],
                        tf.float32,
                    ),
                    pitch,
                    tid,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.n, reshuffle_each_iteration=True)

        ds = ds.batch(self.batch_size, drop_remainder=False)

        # Keras の model.fit() は (inputs, targets) の2要素タプルを期待する。
        # 3要素タプル (audio, pitch, timbre_id) をそのまま渡すと
        # audio=inputs / pitch=targets と誤解釈されてしまう。
        # train_step が data をまるごと受け取れるよう
        # ((audio, pitch, timbre_id), dummy) の形にラップする。
        ds = ds.map(
            lambda audio, pitch, tid: ((audio, pitch, tid), tf.constant(0)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


# ============================================================
# コールバック
# ============================================================
class CheckpointCallback(tf.keras.callbacks.Callback):
    """
    エポックごとに最新重みを保存し、
    val_loss が改善した場合はベスト重みも保存する。
    """

    def __init__(self, ckpt_dir: str):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.best_path = self.ckpt_dir / "best_weights.weights.h5"
        self.latest_path = self.ckpt_dir / "latest.weights.h5"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 毎エポック: 最新重みを保存
        self.model.save_weights(str(self.latest_path))

        # val_loss が改善した場合: ベスト重みを保存
        val_loss = logs.get("val_loss", float("inf"))
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.model.save_weights(str(self.best_path))
            print(
                f"  → ベスト更新 val_loss={val_loss:.4f}  saved: {self.best_path}"
            )


class PrintMetricsCallback(tf.keras.callbacks.Callback):
    """
    train_step が返すカスタムメトリクスを整形して表示するコールバック。
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        items = [
            f"loss={logs.get('loss', 0):.4f}",
            f"recon={logs.get('recon', 0):.4f}",
            f"kl={logs.get('kl', 0):.4f}",
            f"kl_w={logs.get('kl_weight', 0):.5f}",
            f"stft={logs.get('stft', 0):.4f}",
            f"mel={logs.get('mel', 0):.4f}",
            f"z_std={logs.get('z_std_ema', 0):.3f}",
            f"g_norm={logs.get('grad_norm', 0):.3f}",
        ]
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            items.append(f"val_loss={val_loss:.4f}")
        print(f"  Epoch {epoch+1:4d}  " + "  ".join(items))


# ============================================================
# メイン: build_and_train
# ============================================================
def build_and_train(cfg: dict):
    # --- 再現性 ---
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # --- GPU メモリ設定 (動的確保) ---
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # -----------------------------------------------------------------
    # データ準備
    # -----------------------------------------------------------------
    print("=== データセットを読み込み中 ===")
    samples = parse_csv(cfg["csv"], cfg["wav_root"])
    print(f"  総サンプル数: {len(samples)}")

    # 音色ごとのサンプル数を表示
    for name, tid in TIMBRE_MAP.items():
        n = sum(1 for s in samples if s["timbre_id"] == tid)
        print(f"  {name} (id={tid}): {n} サンプル")

    # シャッフルして train / val に分割
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * cfg["val_split"]))
    val_s = samples[:n_val]
    train_s = samples[n_val:]
    print(f"  train: {len(train_s)}  val: {len(val_s)}")

    print("\n=== 学習データをロード中 ===")
    train_ds = AudioDataset(train_s, cfg["batch_size"], shuffle=True).build()
    print("=== 検証データをロード中 ===")
    val_ds = AudioDataset(val_s, cfg["batch_size"], shuffle=False).build()

    # steps_per_epoch の計算
    steps_per_epoch = (len(train_s) + cfg["batch_size"] - 1) // cfg[
        "batch_size"
    ]
    print(f"\n  steps_per_epoch: {steps_per_epoch}")

    # -----------------------------------------------------------------
    # モデル構築
    # -----------------------------------------------------------------
    print("\n=== モデルを構築中 ===")
    model = TimeWiseCVAE(steps_per_epoch=steps_per_epoch)

    # ダミー入力でビルド (重みを初期化)
    dummy_audio = tf.zeros([1, TIME_LENGTH, 1])
    dummy_pitch = tf.zeros([1])
    dummy_tid = tf.zeros([1], dtype=tf.int32)
    model((dummy_audio, dummy_pitch, dummy_tid), training=False)
    print(f"  学習パラメータ数: {model.count_params():,}")

    # -----------------------------------------------------------------
    # オプティマイザ・コンパイル
    # -----------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    # train_step / test_step を完全自前実装しているため loss= は不要。
    # run_eagerly=False のままグラフモードで高速に動作する。
    model.compile(optimizer=optimizer, run_eagerly=False)

    # -----------------------------------------------------------------
    # チェックポイントから再開
    # -----------------------------------------------------------------
    ckpt_dir = Path(cfg["ckpt_dir"])
    latest_path = ckpt_dir / "latest.weights.h5"
    if cfg["resume"] and latest_path.exists():
        print(f"\n=== チェックポイントから再開: {latest_path} ===")
        model.load_weights(str(latest_path))
    elif cfg["resume"]:
        print("  チェックポイントが見つかりません。最初から学習します。")

    # -----------------------------------------------------------------
    # コールバック
    # -----------------------------------------------------------------
    callbacks = [
        CheckpointCallback(cfg["ckpt_dir"]),
        PrintMetricsCallback(),
        tf.keras.callbacks.TensorBoard(
            log_dir=cfg["log_dir"],
            histogram_freq=0,
            update_freq="epoch",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # -----------------------------------------------------------------
    # 学習
    # -----------------------------------------------------------------
    print(
        f"\n=== 学習開始  epochs={cfg['epochs']}  batch={cfg['batch_size']} ===\n"
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=0,  # PrintMetricsCallback で代替
    )

    # -----------------------------------------------------------------
    # 最終重みを保存
    # -----------------------------------------------------------------
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "weights_final.weights.h5"
    model.save_weights(str(final_path))
    print(f"\n=== 学習完了 ===")
    print(f"  最終重み  : {final_path}")
    print(f"  ベスト重み: {ckpt_dir / 'best_weights.weights.h5'}")

    # 埋め込みベクトルを表示 (音色の配置確認)
    print("\n=== 学習済み音色埋め込みベクトル ===")
    for name, tid in TIMBRE_MAP.items():
        emb = model.timbre_embedding.embedding(np.array([tid]))[0].numpy()
        formatted = ", ".join(f"{v:+.3f}" for v in emb)
        print(f"  {name:8s} (id={tid}): [{formatted}]")

    return model, history


# ============================================================
# エントリーポイント
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="TimeWiseCVAE 訓練スクリプト")
    p.add_argument(
        "--csv", default=DEFAULTS["csv"], help="データセットCSVのパス"
    )
    p.add_argument(
        "--wav_root",
        default=DEFAULTS["wav_root"],
        help="WAVファイルのルートディレクトリ",
    )
    p.add_argument("--epochs", default=DEFAULTS["epochs"], type=int)
    p.add_argument("--batch_size", default=DEFAULTS["batch_size"], type=int)
    p.add_argument("--lr", default=DEFAULTS["lr"], type=float)
    p.add_argument("--val_split", default=DEFAULTS["val_split"], type=float)
    p.add_argument("--seed", default=DEFAULTS["seed"], type=int)
    p.add_argument(
        "--out_dir",
        default=DEFAULTS["out_dir"],
        help="最終重みの出力先ディレクトリ",
    )
    p.add_argument(
        "--ckpt_dir",
        default=DEFAULTS["ckpt_dir"],
        help="チェックポイントディレクトリ",
    )
    p.add_argument(
        "--log_dir",
        default=DEFAULTS["log_dir"],
        help="TensorBoardログディレクトリ",
    )
    p.add_argument(
        "--resume", action="store_true", help="latest.weights.h5 から再開"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = vars(args)
    build_and_train(cfg)
