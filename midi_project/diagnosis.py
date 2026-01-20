import tensorflow as tf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from model import TimeWiseCVAE, LATENT_DIM, sample_z
from create_datasets import MAX_LEN, make_dataset_from_synth_csv

# モデルロード
print("=" * 60)
print("根本原因診断")
print("=" * 60)

model = TimeWiseCVAE()
dummy_cond = tf.zeros((1, 4), dtype=tf.float32)
dummy_x = tf.zeros((1, MAX_LEN), dtype=tf.float32)
_ = model([dummy_x, dummy_cond], training=False)
model.load_weights("weights/best_model.weights.h5")
print("✓ モデルをロードしました\n")


# ========================================
# 診断1: 訓練データの分析
# ========================================
print("\n" + "=" * 60)
print("診断1: 訓練データの音色バランス")
print("=" * 60)

try:
    import pandas as pd

    df = pd.read_csv("dataset.csv")

    timbre_counts = df["timbre"].value_counts()
    print("\n音色ごとのサンプル数:")
    for timbre, count in timbre_counts.items():
        print(f"  {timbre}: {count} サンプル")

    total = len(df)
    print(f"\n合計: {total} サンプル")

    print("\n比率:")
    for timbre, count in timbre_counts.items():
        ratio = count / total * 100
        print(f"  {timbre}: {ratio:.1f}%")

    # 不均衡チェック
    max_count = timbre_counts.max()
    min_count = timbre_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n不均衡比: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("⚠️  警告: 音色間のデータ不均衡が大きいです")
        print("   → これがノイズ問題の原因の可能性があります")

except Exception as e:
    print(f"dataset.csv の読み込みエラー: {e}")


# ========================================
# 診断2: 訓練データの実音声を確認
# ========================================
print("\n" + "=" * 60)
print("診断2: 訓練データの実音声分析")
print("=" * 60)

try:
    # データセットから実際の音声を取得
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=1)
    dataset = dataset.take(30)  # 最初の30サンプル

    timbre_stats = {
        "screech": {"rms": [], "noise_ratio": []},
        "acid": {"rms": [], "noise_ratio": []},
        "pluck": {"rms": [], "noise_ratio": []},
    }

    timbre_map = {0: "screech", 1: "acid", 2: "pluck"}

    for x, cond in dataset:
        audio = x.numpy().squeeze()
        timbre_onehot = cond.numpy()[0, 1:]
        timbre_idx = np.argmax(timbre_onehot)
        timbre_name = timbre_map[timbre_idx]

        # RMS計算
        rms = np.sqrt(np.mean(audio**2))

        # ノイズ比率推定（高周波成分の比率）
        fft = np.fft.rfft(audio)
        freq = np.fft.rfftfreq(len(audio), 1 / 48000)

        low_freq_power = np.sum(np.abs(fft[freq < 2000]) ** 2)
        high_freq_power = np.sum(np.abs(fft[freq > 8000]) ** 2)
        total_power = np.sum(np.abs(fft) ** 2)

        noise_ratio = high_freq_power / (total_power + 1e-8)

        timbre_stats[timbre_name]["rms"].append(rms)
        timbre_stats[timbre_name]["noise_ratio"].append(noise_ratio)

    print("\n訓練データの統計:")
    for timbre_name, stats in timbre_stats.items():
        if len(stats["rms"]) > 0:
            avg_rms = np.mean(stats["rms"])
            avg_noise = np.mean(stats["noise_ratio"])
            print(f"\n{timbre_name.upper()}:")
            print(f"  平均RMS: {avg_rms:.4f}")
            print(f"  平均ノイズ比: {avg_noise:.4f}")

            if avg_noise > 0.3:
                print(f"  ⚠️  警告: 訓練データ自体にノイズが多い可能性")

except Exception as e:
    print(f"データセット分析エラー: {e}")


# ========================================
# 診断3: Prior Networkの出力分析
# ========================================
print("\n" + "=" * 60)
print("診断3: Prior Networkの音色ごとの挙動")
print("=" * 60)


def midi_to_normalized_pitch(midi_note):
    min_midi = 24
    max_midi = 96
    return (midi_note - min_midi) / (max_midi - min_midi)


timbre_dict = {"screech": [1, 0, 0], "acid": [0, 1, 0], "pluck": [0, 0, 1]}

print("\n各音色でのPrior Networkの出力:")
for timbre_name, timbre_vec in timbre_dict.items():
    print(f"\n{timbre_name.upper()}:")

    # 複数のピッチで確認
    all_means = []
    all_logvars = []

    for midi in [36, 48, 60, 72]:  # C2, C3, C4, C5
        pitch = midi_to_normalized_pitch(midi)
        cond = tf.constant([[pitch] + timbre_vec], dtype=tf.float32)

        prior_mean, prior_logvar = model.prior_net(cond)
        all_means.append(prior_mean.numpy())
        all_logvars.append(prior_logvar.numpy())

    all_means = np.concatenate(all_means, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)

    print(f"  Mean range: [{all_means.min():.4f}, {all_means.max():.4f}]")
    print(f"  Mean std: {all_means.std():.4f}")
    print(f"  Logvar range: [{all_logvars.min():.4f}, {all_logvars.max():.4f}]")
    print(f"  Logvar std: {all_logvars.std():.4f}")

    # ★重要: logvar stdが小さすぎる場合
    if all_logvars.std() < 0.1:
        print(f"  ⚠️  Prior Networkがほとんど学習できていません")
        print(f"     → この音色はPrior Networkを使えていない可能性")


# ========================================
# 診断4: Decoderの各コンポーネント出力
# ========================================
print("\n" + "=" * 60)
print("診断4: Decoderのコンポーネント別出力")
print("=" * 60)

for timbre_name in ["screech", "acid", "pluck"]:
    print(f"\n{timbre_name.upper()}:")

    pitch = midi_to_normalized_pitch(48)
    cond = tf.constant([[pitch] + timbre_dict[timbre_name]], dtype=tf.float32)

    prior_mean, prior_logvar = model.prior_net(cond)
    z = prior_mean

    # ★重要: NoiseGeneratorの出力を直接確認
    # model.pyのNoiseGeneratorにアクセス
    try:
        # Decoderの内部レイヤーを探す
        noise_gen = None
        for layer in model.decoder.layers:
            if hasattr(layer, "name") and "noise" in layer.name.lower():
                noise_gen = layer
                break

        if noise_gen is None:
            # 手動でNoiseGeneratorを作成して確認
            from model import NoiseGenerator

            noise_gen = NoiseGenerator()
            # Decoderの重みをコピー（簡易版）

        noise_output = noise_gen(z, cond)
        noise_rms = tf.sqrt(tf.reduce_mean(tf.square(noise_output))).numpy()

        print(f"  Noise component RMS: {noise_rms:.6f}")

        if noise_rms > 0.05:
            print(f"  ⚠️  ノイズ成分が非常に大きいです")
            print(f"     → NoiseGeneratorが過剰に反応している可能性")

    except Exception as e:
        print(f"  Noise分析エラー: {e}")

    # 完全な出力
    full_output = model.decoder([z, cond], training=False)
    full_rms = tf.sqrt(tf.reduce_mean(tf.square(full_output))).numpy()
    print(f"  Full output RMS: {full_rms:.6f}")


# ========================================
# 診断5: Encoderを使った再構成テスト
# ========================================
print("\n" + "=" * 60)
print("診断5: Encoder-Decoderの再構成能力")
print("=" * 60)

try:
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=1)

    timbre_recon_errors = {"screech": [], "acid": [], "pluck": []}

    timbre_map = {0: "screech", 1: "acid", 2: "pluck"}

    for i, (x, cond) in enumerate(dataset.take(30)):
        # Encoderで潜在変数取得
        z_mean, z_logvar = model.encoder(x)

        # Decoderで再構成
        x_recon = model.decoder([z_mean, cond], training=False)

        # 再構成誤差
        recon_error = tf.reduce_mean(tf.square(x - x_recon)).numpy()

        timbre_onehot = cond.numpy()[0, 1:]
        timbre_idx = np.argmax(timbre_onehot)
        timbre_name = timbre_map[timbre_idx]

        timbre_recon_errors[timbre_name].append(recon_error)

    print("\n音色ごとの平均再構成誤差:")
    for timbre_name, errors in timbre_recon_errors.items():
        if len(errors) > 0:
            avg_error = np.mean(errors)
            print(f"  {timbre_name}: {avg_error:.6f}")

            if avg_error > 0.01:
                print(f"    ⚠️  再構成誤差が大きい → この音色の学習が不十分")

except Exception as e:
    print(f"再構成テストエラー: {e}")


# ========================================
# 診断結果のまとめ
# ========================================
print("\n" + "=" * 60)
print("診断結果まとめ")
print("=" * 60)

print(
    """
【確認すべきポイント】

1. データ不均衡
   - screech, acid, pluckのサンプル数が均等か？
   - 不均衡比が2.0以上なら問題

2. 訓練データの品質
   - acidやscreechの訓練データ自体にノイズが多くないか？
   - 平均ノイズ比 > 0.3 なら訓練データが問題

3. Prior Networkの学習不足
   - Logvar stdが0.1以下なら学習できていない
   - 特定の音色だけ学習できていない可能性

4. NoiseGeneratorの過剰反応
   - Noise component RMS > 0.05 なら問題
   - model.pyのNoiseGeneratorの設定を確認

5. Encoder-Decoderの再構成
   - 再構成誤差が音色間で大きく異なるか？
   - acidやscreechだけ誤差が大きいなら学習不足

【次のステップ】

この診断結果に基づいて、以下のいずれかを実施:

A. データ問題の場合
   → dataset.csvを確認し、データを追加/バランス調整

B. NoiseGenerator問題の場合
   → model.pyのNoiseGeneratorのnoise_amount係数を調整

C. Prior Network問題の場合
   → kl_targetをさらに増やして再訓練

D. 学習不足の場合
   → より長いエポック数で再訓練
"""
)

print("=" * 60)
