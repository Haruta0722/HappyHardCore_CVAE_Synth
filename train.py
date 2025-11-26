# file: cvae_time_conditional_improved.py
# TensorFlow 2.x 用（例: 2.10+）
# datasets/labels.csv を読み、input_path, output_path, attack, distortion, thickness, center_tone を使って学習

import os
import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import soundfile as sf
import tensorflow.keras.layers as layers
# -------------------------
# ハイパーパラメータ（調整可）
# -------------------------
SR = 32000
BATCH_SIZE = 4            # 長い波形なら小さめに
LATENT_DIM = 128         # 時間方向の潜在次元（per time-step）
ENC_CHANNELS = [64, 128, 256]
DOWNSAMPLE_FACTORS = [4, 4, 4]   # 合計縮小率 = 64
DEC_CHANNELS = ENC_CHANNELS[::-1]
LEARNING_RATE = 1e-4
EPOCHS = 200
WAVE_L1_WEIGHT = 1.0
STFT_WEIGHT = 1.2
MEL_WEIGHT = 3.0
KL_WEIGHT_MAX = 4.0
ANNEAL_STEPS = 200000    # とてもゆっくり増やす（posterior collapse防止）
FREE_BITS = 0.5          # free bits（nats） per latent-dimension (大きさは調整)
STFT_FFTS = [512, 1024, 2048]
MEL_BINS = 80

# データ CSV パス
LABEL_CSV = "datasets/labels.csv"
BASE_DIR = "."  # CSV内の相対パス基準（必要なら変更）

# -------------------------
# ユーティリティ
# -------------------------
def load_wav(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # RMS normalize to preserve relative amplitude (avoid centering to zero mean causing trivial zero solution)
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.1  # 小さめに揃える（調整可）
    y = y / rms * target_rms
    return y.astype(np.float32)

def write_wav(path, y, sr=SR):
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr)

# -------------------------
# Melフィルタ行列（TFで計算）
# -------------------------
def get_mel_matrix(n_fft, n_mels=MEL_BINS, sr=SR, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )

# -------------------------
# データセット作成
# CSV は input_path, output_path, attack, distortion, thickness, center_tone を持つ
# yields: (input_padded, target_padded), lengths, cond_vector
# -------------------------
def make_dataset_from_csv(csv_path, base_dir=BASE_DIR, batch_size=BATCH_SIZE, shuffle=True):
    df = pd.read_csv(csv_path)
    # 絶対パス化
    df['input_path'] = df['input_path'].apply(lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p)
    df['output_path'] = df['output_path'].apply(lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p)

    # 読み込み generator
    def gen():
        for _, row in df.iterrows():
            x = load_wav(row['input_path'])
            y = load_wav(row['output_path'])
            cond = np.array([row['attack'], row['distortion'], row['thickness'], row['center_tone']], dtype=np.float32)
            yield x, y, cond, np.int32(len(x)), np.int32(len(y))

    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=256)
    # expand dims to [T,1]
    ds = ds.map(lambda x, y, c, lx, ly: (tf.expand_dims(x, -1), tf.expand_dims(y, -1), c, lx, ly))
    # padded_batch（inputとoutputそれぞれ可変長）
    ds = ds.padded_batch(batch_size,
                         padded_shapes=([None,1], [None,1], [4], (), ()),
                         padding_values=(0.0, 0.0, 0.0, 0, 0))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Encoder / Decoder の構築（条件を使う）
# Encoder: input waveform -> downsample convs -> mean/logvar (time-distributed)
# Decoder: z(t) + cond -> upsample convs -> waveform
# 条件は FiLM 風に各デコーダブロックへ注入する
# -------------------------
class FiLM(layers.Layer):
    def __init__(self, channels, cond_dim=4, name=None):
        super().__init__(name=name)
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        # cond 表現を強化する MLP
        self.net = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.channels * 2,
                         kernel_initializer="zeros",
                         bias_initializer="zeros")  # 最初は変化させない
        ])

    def call(self, inputs):
        # inputs: [x, cond]
        x, cond = inputs
        gb = self.net(cond)                       # [B, 2*C]
        gamma, beta = tf.split(gb, 2, axis=-1)    # each [B, C]
        gamma = gamma[:, None, :]                 # [B,1,C]
        beta  = beta[:, None, :]                  # [B,1,C]
        return x * (1.0 + gamma) + beta

def build_encoder(latent_dim=LATENT_DIM, channels=ENC_CHANNELS, down_factors=DOWNSAMPLE_FACTORS):
    inp = layers.Input(shape=(None,1), name='enc_input')
    x = inp
    for i, (ch, f) in enumerate(zip(channels, down_factors)):
        x = layers.Conv1D(ch, kernel_size=9, strides=1, padding='same', activation='relu', name=f'enc_conv_{i}_a')(x)
        x = layers.Conv1D(ch, kernel_size=9, strides=f, padding='same', activation='relu', name=f'enc_conv_{i}_b')(x)
        x = layers.BatchNormalization(name=f'enc_bn_{i}')(x)
    # project to mean/logvar (time-distributed)
    mean = layers.Conv1D(latent_dim, kernel_size=3, padding='same', name='enc_mean')(x)
    logvar = layers.Conv1D(latent_dim, kernel_size=3, padding='same', name='enc_logvar')(x)
    return tf.keras.Model(inp, [mean, logvar], name='time_encoder')

def residual_block(x, channels, dilation=1, name=None):
    """Stable dilated residual block.
       Conv -> ReLU -> Conv(zeros) -> scale -> add."""
    
    # 1: dilated conv
    h = layers.Conv1D(
        filters=channels,
        kernel_size=3,
        dilation_rate=dilation,
        padding="same",
        activation=None,
        name=f"{name}_dilconv"
    )(x)
    h = layers.Activation("relu", name=f"{name}_dil_relu")(h)

    # 2: final conv with ZERO initialization
    h = layers.Conv1D(
        filters=channels,
        kernel_size=1,
        padding="same",
        activation=None,
        name=f"{name}_proj",
        kernel_initializer="zeros",
        bias_initializer="zeros"
    )(h)

    # 3: residual scaling (0.1 is standard)
    h = layers.Lambda(lambda t: 0.1 * t, name=f"{name}_scale")(h)

    # 4: skip add (NO ReLU here)
    out = layers.Add(name=f"{name}_add")([x, h])

    return out

# -------------------------
# Decoder 全体（差し替え用）
# -------------------------
def build_decoder(
    latent_dim=128,
    cond_dim=4,
    channels=[256, 128, 64],     # DEC_CHANNELS 相当
    up_factors=[4, 4, 4],        # 合計 64
    n_dilated_per_stage=6        # 各アップ段に入れる dilated 層数（変更可）
):
    z_in = layers.Input(shape=(None, latent_dim), name="dec_z_in")
    cond_in = layers.Input(shape=(cond_dim,), name="dec_cond_in")

    # ----- 入力プロジェクション -----
    x = layers.Conv1D(channels[0], kernel_size=3, padding="same", activation=None, name="dec_proj")(z_in)
    x = FiLM(channels[0], cond_dim=cond_dim, name="dec_proj_film")([x, cond_in])
    x = layers.Activation("relu", name="dec_proj_relu")(x)

    # ----- 各アップサンプルブロック -----
    for stage_idx, factor in enumerate(up_factors):
        ch = channels[min(stage_idx, len(channels) - 1)]
        x = layers.Conv1DTranspose(filters=ch, kernel_size=factor*2, strides=factor, padding="same")(x)
        x = FiLM(ch, cond_dim=cond_dim, name=f"dec_up_film_s{stage_idx}")([x, cond_in])
        x = layers.Activation("relu", name=f"dec_up_relu_s{stage_idx}")(x)

        # dilated residual stack (各ステージごとに複数)
        # dilations: 1,2,4,8,... を使う（n_dilated_per_stage 個）
        for i in range(n_dilated_per_stage):
            d = 2 ** (i % 6)  # 1,2,4,8,16,32 を繰り返す（必要なら変える）
            x = residual_block(x, channels=ch, dilation=d, name=f"dec_stage{stage_idx}_res{i}")

        # stag-end にもう一回 FiLM して安定化
        x = FiLM(ch, cond_dim=cond_dim, name=f"dec_stage{stage_idx}_postfilm")([x, cond_in])
        x = layers.Activation("relu", name=f"dec_stage{stage_idx}_post_relu")(x)

    # ----- 出力プロジェクション -----
    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu", name="dec_final_conv1")(x)
    out = layers.Conv1D(1, kernel_size=7, padding="same", activation="tanh", name="dec_final_out")(x)
    # NaN/Inf safe guard を Lambda で包む

    return tf.keras.Model([z_in, cond_in], out, name="time_decoder_improved")
# -------------------------
# Reparameterize（time-distributed）
# -------------------------
class ReparamTime(layers.Layer):
    def call(self, mean, logvar, training=True):
        if training:
            eps = tf.random.normal(tf.shape(mean))
            return mean + tf.exp(0.5 * logvar) * eps
        else:
            return mean  # use mean for deterministic inference

# -------------------------
# Loss utilities: complex STFT L1, log-mag L1, mel L1
# -------------------------
import tensorflow as tf

# -----------------------------------
# 安全な complex STFT L1
# -----------------------------------
def stft_complex_l1(y, y_hat, fft_size, hop_size, eps=1e-7):
    S = tf.signal.stft(y, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size)
    S_hat = tf.signal.stft(y_hat, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size)

    # magnitude only
    S_mag = tf.maximum(tf.abs(S), eps)
    S_hat_mag = tf.maximum(tf.abs(S_hat), eps)

    return tf.reduce_mean(tf.abs(S_mag - S_hat_mag))


def log_mag_l1(y, y_hat, fft_size, hop_size, eps=1e-7):
    S = tf.signal.stft(y, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size)
    S_hat = tf.signal.stft(y_hat, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size)

    S_mag = tf.maximum(tf.abs(S), eps)
    S_hat_mag = tf.maximum(tf.abs(S_hat), eps)

    return tf.reduce_mean(tf.abs(tf.math.log(S_mag) - tf.math.log(S_hat_mag)))


def mel_l1(y, y_hat, n_fft=1024, hop=256, n_mels=MEL_BINS, eps=1e-7):
    S = tf.signal.stft(y, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    S_hat = tf.signal.stft(y_hat, frame_length=n_fft, frame_step=hop, fft_length=n_fft)

    S_mag = tf.maximum(tf.abs(S), eps)
    S_hat_mag = tf.maximum(tf.abs(S_hat), eps)

    mel_mat = tf.cast(get_mel_matrix(n_fft, n_mels=n_mels), S_mag.dtype)

    mel = tf.matmul(S_mag, mel_mat)
    mel_hat = tf.matmul(S_hat_mag, mel_mat)

    mel = tf.maximum(mel, eps)
    mel_hat = tf.maximum(mel_hat, eps)

    return tf.reduce_mean(tf.abs(tf.math.log(mel) - tf.math.log(mel_hat)))

# -------------------------
# CVAE 主クラス（カスタム train_step）
# -------------------------
class WaveTimeConditionalCVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, latent_dim=LATENT_DIM,
                 kl_anneal_steps=ANNEAL_STEPS, kl_weight_max=KL_WEIGHT_MAX,
                 free_bits=FREE_BITS, steps_per_epoch = 243, initial_epoch = 0,
                 wave_weight=WAVE_L1_WEIGHT, stft_weight=STFT_WEIGHT, mel_weight=MEL_WEIGHT):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reparam = ReparamTime()
        self.latent_dim = latent_dim
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_weight_max = kl_weight_max
        self.free_bits = free_bits
        self.wave_weight = wave_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        # total_stepsをtf.Variableで管理
        initial_steps = initial_epoch * (steps_per_epoch or 0)
        self.total_steps = tf.Variable(initial_steps, dtype=tf.int64, trainable=False, name="total_steps")

        print(f"[info] total_steps 初期化: {int(self.total_steps.numpy())}")

        # メトリクス
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon")
        self.stft_tracker = tf.keras.metrics.Mean(name="stft")
        self.mel_tracker = tf.keras.metrics.Mean(name="mel")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")
        self.mean_mu_tracker = tf.keras.metrics.Mean(name="mean_mu")
        self.mean_logvar_tracker = tf.keras.metrics.Mean(name="mean_logvar")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.stft_tracker,
                self.mel_tracker, self.kl_tracker, self.mean_mu_tracker, self.mean_logvar_tracker]

    def kl_weight(self):
        step = tf.cast(self.total_steps, tf.float32)
        s = tf.minimum(1.0, step / tf.cast(self.kl_anneal_steps, tf.float32))
        return s * self.kl_weight_max 
    
    def call(self, inputs, training=False):
            # 柔軟に入力数を判定
        if len(inputs) == 5:
            x, _, cond, lx, ly = inputs
        elif len(inputs) == 3:
            # outputは無視、長さ情報はダミー
            x, cond, _ = inputs
            lx = ly = None
        elif len(inputs) == 2:
            x, cond = inputs
            lx = ly = None
        else:
            raise ValueError(f"Unexpected number of inputs: {len(inputs)}")

        mean, logvar = self.encoder(x, training=training)
        z = self.reparam(mean, logvar, training=training)
        y_hat = self.decoder([z, cond], training=training)
        return y_hat

    def train_step(self, data):
        # data: (x_padded, y_padded, cond, len_x, len_y)
        x, y, cond, len_x, len_y = data
        batch_size = tf.shape(x)[0]
        maxlen_x = tf.shape(x)[1]
        maxlen_y = tf.shape(y)[1]

        # 元の mask（最大長で作られているが後で min_len に合わせる）
        mask_x = tf.cast(tf.expand_dims(tf.sequence_mask(len_x, maxlen=maxlen_x), -1), tf.float32)
        mask_y_full = tf.cast(tf.expand_dims(tf.sequence_mask(len_y, maxlen=maxlen_y), -1), tf.float32)

        with tf.GradientTape() as tape:
            mean, logvar = self.encoder(x, training=True)  # [B, Tz, D]
            z = self.reparam(mean, logvar, training=True)  # [B, Tz, D]
            y_hat = self.decoder([z, cond], training=True)  # [B, T_hat, 1]

            # --- 重要: y と y_hat の時間次元を揃える ---
            len_y_hat = tf.shape(y_hat)[1]
            min_len = tf.minimum(maxlen_y, len_y_hat)
            # 切り詰め（両方とも min_len に揃える）
            y_trim = y[:, :min_len, :]
            yhat_trim = y_hat[:, :min_len, :]
            # mask を min_len に合わせる
            mask_y = tf.cast(tf.expand_dims(tf.sequence_mask(len_y, maxlen=min_len), -1), tf.float32)
            # ------------------------------------------------

            # waveform L1 (mask target length)
            wave_diff = tf.abs((y_trim - yhat_trim) * mask_y)
            recon_loss = tf.reduce_sum(wave_diff) / (tf.reduce_sum(mask_y) + 1e-9)

            # STFT + complex losses: use squeeze to [B, T]
            y_s = tf.squeeze(y_trim * mask_y, -1)
            yhat_s = tf.squeeze(yhat_trim * mask_y, -1)
            stft_terms = []
            for n_fft in STFT_FFTS:
                hop = max(64, n_fft // 4)
                stft_terms.append(stft_complex_l1(y_s, yhat_s, n_fft, hop) + log_mag_l1(y_s, yhat_s, n_fft, hop))
            stft_loss = tf.add_n(stft_terms) / float(len(stft_terms))

            # mel loss (single resolution)
            mel_loss = mel_l1(y_s, yhat_s, n_fft=1024, hop=256, n_mels=MEL_BINS)

            # KL per-dim per-time-step
            kl_per = -0.5 * (1.0 + logvar - tf.square(mean) - tf.exp(logvar))  # [B, Tz, D]
            # sum over dim -> [B, Tz]
            kl_time = tf.reduce_sum(kl_per, axis=-1)
            # apply free bits: clip with lower bound FREE_BITS * D
            free_bits_total = self.free_bits * tf.cast(self.latent_dim, tf.float32)
            # we take mean over time and batch after applying free bits
            kl_time_clipped = tf.maximum(kl_time, free_bits_total)
            kl_loss = tf.reduce_mean(kl_time_clipped)

            kl_w = self.kl_weight()
            loss = self.wave_weight * recon_loss + self.stft_weight * stft_loss + self.mel_weight * mel_loss + kl_w * kl_loss 

        grads = tape.gradient(loss, self.trainable_variables)
        # 勾配ノルムクリッピング
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # step 更新とメトリクス
        self.total_steps.assign_add(1)
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.stft_tracker.update_state(stft_loss)
        self.mel_tracker.update_state(mel_loss)
        self.kl_tracker.update_state(kl_loss)
        self.mean_mu_tracker.update_state(tf.reduce_mean(mean))
        self.mean_logvar_tracker.update_state(tf.reduce_mean(logvar))

        return {
            "loss": self.loss_tracker.result(),
            "recon": self.recon_tracker.result(),
            "stft": self.stft_tracker.result(),
            "mel": self.mel_tracker.result(),
            "kl": self.kl_tracker.result(),
            "kl_w": kl_w,
            "mean_mu": self.mean_mu_tracker.result(),
            "mean_logvar": self.mean_logvar_tracker.result(),
        }

    def infer(self, x, cond, len_x):
        # x: [T,1] numpy or tensor -> pad to batch 1 and call encoder+decoder (use mean)
        if isinstance(x, np.ndarray):
            x = tf.expand_dims(x, 0)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 0)
        # build mask if needed
        mean, logvar = self.encoder(x, training=False)
        z = mean  # deterministic
        y_hat = self.decoder([z, cond], training=False)
        return y_hat.numpy()[0, :len_x, 0]

# -------------------------
# モデル作成と学習ループ
# -------------------------
def make_and_train():
    # ===============================
    # データセット構築
    # ===============================
    ds = make_dataset_from_csv(LABEL_CSV, batch_size=BATCH_SIZE)

    # ===============================
    # モデル構築
    # ===============================
    enc = build_encoder(latent_dim=LATENT_DIM, channels=ENC_CHANNELS, down_factors=DOWNSAMPLE_FACTORS)
    dec = build_decoder(latent_dim=LATENT_DIM, channels=DEC_CHANNELS,
                        up_factors=DOWNSAMPLE_FACTORS[::-1], cond_dim=4)
    model = WaveTimeConditionalCVAE(
        enc, dec, latent_dim=LATENT_DIM,
        kl_anneal_steps=ANNEAL_STEPS, kl_weight_max=KL_WEIGHT_MAX, free_bits=FREE_BITS
    )

    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    model.compile(optimizer=opt)

    # ===============================
    # チェックポイント設定
    # ===============================
    ckpt_dir = "checkpoints_cvae"
    os.makedirs(ckpt_dir, exist_ok=True)
       # データセットから1バッチを取ってbuild
    sample_batch = next(iter(ds.take(1)))
    x_sample, y_sample, c_sample, lx, ly = sample_batch
    _ = model([x_sample, c_sample, y_sample])  # ビルドだけ行う

    # 最新の重みファイルを探索

    ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "cvae_*.weights.h5")))
    initial_epoch = 0

    if ckpt_list:
        latest_ckpt = ckpt_list[-1]
        print(f"[info] 最新の重みを読み込みます: {latest_ckpt}")
        model.load_weights(latest_ckpt)

        try:
            initial_epoch = int(os.path.basename(latest_ckpt).split("_")[1].split(".")[0])
            print(f"[info] 再開エポック: {initial_epoch}")
        except Exception as e:
            print(f"[warn] エポック番号の抽出に失敗しました: {e}")
    else:
        print("[info] チェックポイントが見つかりません。新規学習を開始します。")

    # ===============================
    # total_steps の再設定（重要）
    # ===============================
    # 1エポックあたりのステップ数を求める
    steps_per_epoch = 243

    # total_steps が __init__ で定義されている場合のみ実行
    if hasattr(model, "total_steps"):
        model.total_steps.assign(initial_epoch * steps_per_epoch)
        print(f"[info] total_steps を {int(model.total_steps.numpy())} に設定しました")
    else:
        print("[warn] model に total_steps が存在しません。クラス側に tf.Variable を定義してください。")
    # ===============================
    # コールバック設定
    # ===============================
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(ckpt_dir, "cvae_{epoch:03d}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=10, min_lr=1e-7
    )

    class CollapseMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mu = logs.get('mean_mu')
            lv = logs.get('mean_logvar')
            kl = logs.get('kl')
            print(f"[monitor] epoch={epoch:03d} mean_mu={mu:.6f} mean_logvar={lv:.6f} kl(mean)={kl:.6f}")
            if lv is not None and lv < -10.0:
                print("[warning] mean logvar が極端に小さい -> 潜在崩壊の可能性")

    cb_list = [checkpoint_cb, reduce_lr, CollapseMonitor()]

    # ===============================
    # モデルのbuild
    # ===============================
    sample_batch = next(iter(ds.take(1)))
    x_sample, y_sample, c_sample, lx, ly = sample_batch
    _ = model([x_sample, c_sample, y_sample])  # ビルド

    # ===============================
    # 学習開始
    # ===============================
    print("[info] 学習を開始します...")
    model.fit(ds, epochs=EPOCHS, callbacks=cb_list, initial_epoch=initial_epoch)

    # ===============================
    # 最終重み保存
    # ===============================
    final_path = os.path.join(ckpt_dir, "final_weights.h5")
    model.save_weights(final_path)
    print(f"[info] 最終重みを保存しました: {final_path}")

    return model

# -------------------------
# 推論サンプル生成関数
# -------------------------
def transform_single_file(model, input_wav_path, cond_vector, out_path):
    x = load_wav(input_wav_path)
    cond = np.array(cond_vector, dtype=np.float32)
    cond = np.expand_dims(cond, 0)  # batch dim
    x_in = np.expand_dims(x, -1).astype(np.float32)
    y_hat = model.infer(x_in, cond, len(x))
    # 出力を適切な振幅に戻す（もしトレーニングでRMS調整しているなら逆を行う）
    # ここでは単純に出力を -1..1 にクリップして保存
    write_wav(out_path, y_hat, sr=SR)
    print(f"Saved {out_path}")

# -------------------------
# エントリポイント
# -------------------------
if __name__ == "__main__":
    # 学習
    model = make_and_train()

    # 例: 推論（CSVの最初の行をテスト）
    df = pd.read_csv(LABEL_CSV)
    first = df.iloc[0]
    cond = [first['attack'], first['distortion'], first['thickness'], first['center_tone']]
    transform_single_file(model, first['input_path'], cond, out_path="example_transformed.wav")