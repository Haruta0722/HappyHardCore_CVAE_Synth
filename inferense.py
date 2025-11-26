# model_infer_dynamic_build.py
import tensorflow as tf
import numpy as np
import librosa
from train import (
    build_encoder, build_decoder, WaveTimeConditionalCVAE, write_wav
)

def infer (input_wav_name , file_name: str,cond):

    # ====== 音声読み込み ======
    filename =  input_wav_name
    wav, sr = librosa.load(filename, sr=32000)  # SRは学習時と同じに
    wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
    T = wav.shape[1]
    print(f"✅ 読み込み完了: {filename} (長さ: {T} samples)")

    #====== モデル構築 ======
    encoder = build_encoder()
    decoder = build_decoder()
    model = WaveTimeConditionalCVAE(encoder, decoder)

    # ⚡ 実データに基づいてshape確定（これがビルド相当）
    x_in = tf.constant(wav, dtype=tf.float32)
    y_in = tf.zeros_like(x_in)  # ダミー出力（学習時の入力形に合わせる）
    lx = tf.constant([T], dtype=tf.int32)
    ly = tf.constant([T], dtype=tf.int32)
    _ = model((x_in, y_in, cond, lx, ly), training=False)
    print("✅ モデルshape確定（実データベース）")

    # ====== 重みロード ======
    model.load_weights("checkpoints_cvae/cvae_164.weights.h5")
    print("✅ 重みロード完了")

    # ====== 潜在ベクトル推論 ======
    mean, logvar = model.encoder(x_in, training=False)
    eps = tf.random.normal(shape=tf.shape(mean))
    z = mean + tf.exp(0.5 * logvar) * eps * 0.1

    # ====== 再構成 ======
    y_1 = model.decoder([z, cond], training=False)
    y_1 = tf.squeeze(y_1).numpy()
    write_wav(file_name, y_1)
    print("保存完了！")

    return y_1

def conpare_spec(y_1, y_2):
    S_1 = np.abs(librosa.stft(y_1, n_fft=1024, hop_length=256))
    S_2 = np.abs(librosa.stft(y_2, n_fft=1024, hop_length=256))
    print("spec L1:", np.mean(np.abs(np.log(S_1+1e-7)-np.log(S_2+1e-7))))


filename = "datasets/input_data/0010.wav"

y_1 = infer(filename, "y1.wav",cond=tf.constant([[0.0, 0.0, 0.0, 0.0]]) )  # condは適宜変更
y_2 = infer(filename,"y2.wav", cond=tf.constant([[0.0, 0.0, 0.0,  1.0]]))  # condは適宜変更

conpare_spec(y_1, y_2)
