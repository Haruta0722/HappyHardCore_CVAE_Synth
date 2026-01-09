from model import TimeWiseCVAE, TIME_LENGTH, LATENT_DIM, LATENT_STEPS
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from create_datasets import load_wav, crop_or_pad
import soundfile as sf

import sounddevice as sd


# ---------- MIDI note helpers ----------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


SR = 48000

reference = load_wav("datasets/C3/0013.wav")
reference = crop_or_pad(reference, TIME_LENGTH)


# 音声プレビュー用関数
def play_audio(waveform, samplerate=SR):
    sd.stop()  # 連打対策
    sd.play(waveform, samplerate)


def inference_from_reference(
    pitch: float,
    cond: tuple[float, float, float],
    model,
    reference_wave=reference,
    output_name="generated_ref.wav",
):
    """
    ★改善版: 参照音声から潜在変数を抽出して生成
    エンベロープの時間変化もzに反映されるようになった
    """
    print(f"\n[生成] 参照ベース（改善版）")
    print(f"  pitch={pitch} (MIDI), cond={cond}")

    if reference_wave is None:
        t = np.arange(TIME_LENGTH) / 48000.0
        freq = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
        reference_wave = 0.5 * np.sin(2 * np.pi * freq * t)
        reference_wave = reference_wave[:, None].astype(np.float32)
    else:
        reference_wave = np.array(reference_wave, dtype=np.float32)
        if reference_wave.ndim == 1:
            reference_wave = reference_wave[:, None]

    # 条件ベクトル
    pitch_norm = (pitch - 36.0) / 35.0
    cond_vector = tf.constant([[pitch_norm, *cond]], dtype=tf.float32)

    # 参照からzを抽出
    reference_wave_batch = tf.constant(
        reference_wave[None, :, :], dtype=tf.float32
    )
    z_mean, z_logvar = model.encoder([reference_wave_batch, cond_vector])

    # ★重要: 平均値だけでなく、適度なランダム性も加える
    # これにより、エンベロープの微妙な変化も再現される
    z = (
        z_mean
        + tf.exp(0.5 * z_logvar) * tf.random.normal(tf.shape(z_mean)) * 0.1
    )

    # 生成
    x_hat = model.decoder([z, cond_vector])
    x_hat = tf.squeeze(x_hat).numpy()

    # 正規化
    max_val = np.max(np.abs(x_hat))
    if max_val > 1e-6:
        x_hat = x_hat / max_val * 0.95

    sf.write(output_name, x_hat, samplerate=48000)
    print(f"  ✓ 保存: {output_name} (max_amp={max_val:.4f})")

    return x_hat


def midi_to_note_name(midi):
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"


# ---------- GUI ----------
class SynthGUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        root.title("CVAE Synth GUI")

        # === sliders ===
        self.screech = tk.DoubleVar(value=0.0)
        self.acid = tk.DoubleVar(value=0.0)
        self.pluck = tk.DoubleVar(value=0.0)

        for i, (name, var) in enumerate(
            [
                ("screech", self.screech),
                ("acid", self.acid),
                ("pluck", self.pluck),
            ]
        ):
            ttk.Label(root, text=name).grid(row=i, column=0, sticky="w")
            ttk.Scale(root, from_=0.0, to=1.0, variable=var, length=200).grid(
                row=i, column=1
            )

        # === pitch selector ===
        ttk.Label(root, text="pitch").grid(row=3, column=0, sticky="w")

        self.pitch_map = {midi_to_note_name(m): m for m in range(36, 72)}
        self.pitch_var = tk.StringVar(value="C2")

        ttk.OptionMenu(root, self.pitch_var, "C2", *self.pitch_map.keys()).grid(
            row=3, column=1, sticky="w"
        )

        # === generate button ===
        ttk.Button(root, text="Generate", command=self.generate).grid(
            row=4, column=0, columnspan=2, pady=10
        )

        # === waveform plot ===
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.ax.set_title("Waveform")
        self.ax.set_ylim(-1.0, 1.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

    def generate(self):
        pitch_name = self.pitch_var.get()
        pitch = self.pitch_map[pitch_name]

        cond = (
            self.screech.get(),
            self.acid.get(),
            self.pluck.get(),
        )

        waveform = inference_from_reference(
            pitch=pitch,
            cond=cond,
            model=self.model,
            output_name="generated_ref.wav",
        )

        # --- 再生 ---
        play_audio(waveform)

        # --- 波形描画 ---
        self.ax.clear()
        self.ax.set_title(f"{pitch_name}  cond={cond}")
        self.ax.plot(waveform)
        self.ax.set_ylim(-1.0, 1.0)
        self.canvas.draw()


# ---------- main ----------
if __name__ == "__main__":
    root = tk.Tk()

    print("=" * 60)
    print("DDSP風モデル 推論スクリプト")
    print("=" * 60)

    # モデル読み込み
    print("\n[1] モデル読み込み中...")
    model = TimeWiseCVAE()

    # ダミーデータでビルド
    dummy_x = tf.zeros((1, TIME_LENGTH, 1))
    dummy_cond = tf.zeros((1, 4))
    _ = model((dummy_x, dummy_cond), training=False)

    ckpt_path = "weights/epoch_137.weights.h5"
    model.load_weights(ckpt_path)
    print(f"✓ モデルの重みを読み込みました: {ckpt_path}")

    app = SynthGUI(root, model)
    root.mainloop()
