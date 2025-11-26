import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import soundfile as sf

from train import (
    build_encoder, build_decoder,
    WaveTimeConditionalCVAE
)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

SR = 32000
CKPT = "checkpoints_cvae/cvae_164.weights.h5"


st.set_page_config(layout="wide")
st.title("WaveTime Conditional CVAE – Inference UI")

# ===== UI =====
uploaded_file = st.file_uploader("音声ファイルをアップロード (.wav)", type=["wav"])

cond_attack = st.slider("Attack", 0.0, 1.0, 0.0)
cond_dist   = st.slider("Distortion", 0.0, 1.0, 0.0)
cond_thick  = st.slider("Thickness", 0.0, 1.0, 0.0)
cond_center = st.slider("Center", 0.0, 1.0, 0.0)

cond = tf.constant([[cond_attack, cond_dist, cond_thick, cond_center]], dtype=tf.float32)

run_button = st.button("推論実行")

# ===== 処理 =====
if uploaded_file and run_button:
    with st.spinner("音声読み込み中..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        wav, sr = librosa.load(tmp_path, sr=SR)
        wav = np.expand_dims(wav, axis=[0, -1])  # [1, T, 1]
        T = wav.shape[1]

        st.success(f"Loaded audio (T={T})")

    # ===== モデル構築 =====
    with st.spinner("モデル構築 & ビルド中..."):
        encoder = build_encoder()
        decoder = build_decoder()
        model = WaveTimeConditionalCVAE(encoder, decoder)

        x_in = tf.constant(wav, dtype=tf.float32)
        y_in = tf.zeros_like(x_in)
        lx = tf.constant([T], dtype=tf.int32)
        ly = tf.constant([T], dtype=tf.int32)

        _ = model((x_in, y_in, cond, lx, ly), training=False)
        model.load_weights(CKPT)

    st.success("モデル準備完了")

    # ===== 推論 =====
    with st.spinner("推論中..."):
        mean, logvar = model.encoder(x_in, training=False)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(0.5 * logvar) * eps * 0.1
        y_hat = model.decoder([z, cond], training=False)
        y_hat = tf.squeeze(y_hat).numpy()

    # ===== 出力 =====
    st.subheader("Input Audio")
    st.audio(wav.squeeze(), sample_rate=SR)

    st.subheader("Generated Audio")
    st.audio(y_hat, sample_rate=SR)

    sf.write("output.wav", y_hat, SR)
    st.download_button(
        "生成音声をダウンロード",
        data=open("output.wav", "rb"),
        file_name="generated.wav"
    )