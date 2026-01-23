# test_training.py
from model import TimeWiseCVAE, TIME_LENGTH
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf
import os

# データセット
dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=2)
dataset = dataset.repeat()

# モデル
model = TimeWiseCVAE(steps_per_epoch=10)

# ビルド
dummy_x = tf.zeros((1, TIME_LENGTH, 1))
dummy_cond = tf.zeros((1, 4))
_ = model((dummy_x, dummy_cond), training=False)

# パラメータ数確認
total = sum([tf.size(v).numpy() for v in model.trainable_variables])
print(f"総パラメータ数: {total:,}")

# コンパイル
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

# 1エポックだけ訓練
print("\n1エポックだけ訓練してテスト...")
model.fit(dataset, epochs=1, steps_per_epoch=10)

# 保存
os.makedirs("test_checkpoints", exist_ok=True)
model.save_weights("test_checkpoints/test.weights.h5")
print("\n✓ 保存成功: test_checkpoints/test.weights.h5")

# サイズ確認
size = os.path.getsize("test_checkpoints/test.weights.h5")
print(f"ファイルサイズ: {size:,} bytes ({size/1024/1024:.2f} MB)")

if size < 1024:  # 1KB以下
    print("❌ ファイルが小さすぎます！保存に失敗しています")
else:
    print("✅ 正常なサイズです")