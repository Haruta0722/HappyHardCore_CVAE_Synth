"""
CVAE Synth GUI — Python / tkinter
鍵盤を押すとそのピッチで即座に生成・再生する。

依存:
  pip install sounddevice soundfile numpy tensorflow
"""

import tkinter as tk
import math
import threading
import numpy as np

# ── サウンド関連 ───────────────────────────────────────────────────────
try:
    import sounddevice as sd

    HAS_SD = True
except ImportError:
    HAS_SD = False
    print("[警告] sounddevice が見つかりません。再生機能は無効です。")

try:
    import soundfile as sf

    HAS_SF = True
except ImportError:
    HAS_SF = False

# ── モデル関連 ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from model import TimeWiseCVAE, TIME_LENGTH, LATENT_DIM, LATENT_STEPS
    from create_datasets import load_wav, crop_or_pad

    HAS_MODEL = True
except Exception as e:
    HAS_MODEL = False
    TIME_LENGTH = 48000
    print(f"[警告] モデル読み込みスキップ: {e}")

# ══════════════════════════════════════════════════════════════════════
#  定数 / カラー
# ══════════════════════════════════════════════════════════════════════
SR = 48000

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHITE_PCS = [0, 2, 4, 5, 7, 9, 11]
BLACK_PCS = {1, 3, 6, 8, 10}
MIDI_MIN, MIDI_MAX = 36, 71

BG = "#0d0d0d"
PANEL = "#161616"
PANEL2 = "#1c1c1c"
BORDER = "#2a2a2a"
ACCENT = "#e8ff00"
ACCENT2 = "#ff5500"
ACCENT3 = "#00d4ff"
ACID_COL = "#aaff00"
PITCH_COL = "#cc44ff"
DIM = "#555555"
KEY_WHITE = "#e8e8e8"
KEY_BLACK = "#1a1a1a"
KEY_ACT = ACCENT

KNOB_COLORS = {"screech": ACCENT2, "acid": ACID_COL, "pluck": ACCENT3}


# ══════════════════════════════════════════════════════════════════════
#  ヘルパー
# ══════════════════════════════════════════════════════════════════════
def midi_to_note_name(m):
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pitch_norm(m):
    """(pitch - 36) / 35"""
    return (m - 36) / 35.0


# ══════════════════════════════════════════════════════════════════════
#  ノブウィジェット
# ══════════════════════════════════════════════════════════════════════
class Knob(tk.Canvas):
    START_DEG = 225
    SWEEP = 270

    def __init__(self, parent, label, color, command=None, size=72, **kw):
        super().__init__(
            parent,
            width=size,
            height=size + 32,
            bg=PANEL,
            highlightthickness=0,
            **kw,
        )
        self.size = size
        self.color = color
        self.command = command
        self.label = label
        self._value = 0.0
        self._drag_y = None
        self._drag_v = None

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)
        self._draw()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = max(0.0, min(1.0, v))
        self._draw()
        if self.command:
            self.command(self._value)

    def _draw(self):
        self.delete("all")
        s = self.size
        cx = cy = s // 2
        r_outer = s // 2 - 4
        r_track = r_outer - 7
        r_ind = r_outer - 12

        self.create_oval(
            cx - r_outer,
            cy - r_outer,
            cx + r_outer,
            cy + r_outer,
            fill="#111",
            outline=BORDER,
            width=1.5,
        )
        self.create_oval(
            cx - r_track - 4,
            cy - r_track - 4,
            cx + r_track + 4,
            cy + r_track + 4,
            fill="#1e1e1e",
            outline="#2e2e2e",
            width=1,
        )
        self._arc(cx, cy, r_track, 0, 1.0, "#2a2a2a", width=4)
        if self._value > 0.005:
            self._arc(cx, cy, r_track, 0, self._value, self.color, width=4)

        angle = math.radians(self.START_DEG + self.SWEEP * self._value - 90)
        ix = cx + r_ind * math.cos(angle)
        iy = cy + r_ind * math.sin(angle)
        self.create_line(
            cx, cy, ix, iy, fill=self.color, width=2.5, capstyle=tk.ROUND
        )
        self.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3, fill="#444", outline=""
        )

        self.create_text(
            s // 2,
            s + 8,
            text=self.label,
            fill=DIM,
            font=("Courier", 8),
            anchor="n",
        )
        self.create_text(
            s // 2,
            s + 20,
            text=f"{self._value:.2f}",
            fill=self.color,
            font=("Courier", 9, "bold"),
            anchor="n",
        )

    def _arc(self, cx, cy, r, v0, v1, color, width=3):
        a0 = math.radians(self.START_DEG + self.SWEEP * v0 - 90)
        a1 = math.radians(self.START_DEG + self.SWEEP * v1 - 90)
        steps = max(2, int(abs(v1 - v0) * 60))
        pts = []
        for i in range(steps + 1):
            t = i / steps
            a = a0 + (a1 - a0) * t
            pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
        if len(pts) >= 4:
            self.create_line(
                *pts, fill=color, width=width, smooth=True, capstyle=tk.ROUND
            )

    def _on_press(self, e):
        self._drag_y = e.y_root
        self._drag_v = self._value

    def _on_drag(self, e):
        if self._drag_y is None:
            return
        self.value = self._drag_v + (self._drag_y - e.y_root) / 160

    def _on_release(self, _):
        self._drag_y = None

    def _on_wheel(self, e):
        delta = getattr(e, "delta", 0)
        if e.num == 4:
            delta = 120
        if e.num == 5:
            delta = -120
        self.value = self._value + delta / 12000


# ══════════════════════════════════════════════════════════════════════
#  ピアノ鍵盤
# ══════════════════════════════════════════════════════════════════════
class PianoKeyboard(tk.Canvas):
    WW, WH = 30, 90
    BW, BH = 19, 54

    def __init__(self, parent, on_note, **kw):
        self._white_keys = [
            m for m in range(MIDI_MIN, MIDI_MAX + 1) if m % 12 in WHITE_PCS
        ]
        total_w = len(self._white_keys) * self.WW + 2
        super().__init__(
            parent,
            width=total_w,
            height=self.WH + 4,
            bg=BG,
            highlightthickness=0,
            **kw,
        )
        self.on_note = on_note
        self._active = MIDI_MIN
        self._key_items = {}
        self._build()
        self.bind("<ButtonPress-1>", self._click)

    def _build(self):
        white_x = {}
        wi = 0
        for m in range(MIDI_MIN, MIDI_MAX + 1):
            if m % 12 in WHITE_PCS:
                white_x[m] = wi * self.WW + 1
                wi += 1

        for m, x in white_x.items():
            ids = []
            rect = self.create_rectangle(
                x,
                1,
                x + self.WW - 1,
                self.WH,
                fill=KEY_WHITE,
                outline="#888",
                width=1,
                tags=f"key{m}",
            )
            ids.append(rect)
            if m % 12 == 0:
                txt = self.create_text(
                    x + self.WW // 2,
                    self.WH - 10,
                    text=midi_to_note_name(m),
                    font=("Courier", 6),
                    fill="#666",
                    tags=f"key{m}",
                )
                ids.append(txt)
            self._key_items[m] = ids

        for m in range(MIDI_MIN, MIDI_MAX + 1):
            if m % 12 not in BLACK_PCS:
                continue
            left = m - 1
            while left % 12 not in WHITE_PCS:
                left -= 1
            x = white_x[left] + self.WW - self.BW // 2
            rect = self.create_rectangle(
                x,
                0,
                x + self.BW,
                self.BH,
                fill=KEY_BLACK,
                outline="#000",
                width=1,
                tags=f"key{m}",
            )
            self._key_items[m] = [rect]

        self._highlight(self._active)

    def _click(self, e):
        items = self.find_overlapping(e.x - 1, e.y - 1, e.x + 1, e.y + 1)
        hit_midi = None
        for item in reversed(items):
            for tag in self.gettags(item):
                if tag.startswith("key"):
                    try:
                        midi = int(tag[3:])
                        if midi in self._key_items:
                            hit_midi = midi
                            break
                    except ValueError:
                        pass
            if hit_midi is not None:
                break
        if hit_midi is not None:
            self.select(hit_midi)

    def select(self, midi):
        old = self._active
        self._active = midi
        self._unhighlight(old)
        self._highlight(midi)
        self.on_note(midi)  # ← SynthApp._on_note_and_generate を呼ぶ

    def _highlight(self, midi):
        if midi not in self._key_items:
            return
        color = KEY_ACT
        for item in self._key_items[midi]:
            try:
                self.itemconfig(item, fill=color)
            except:
                pass

    def _unhighlight(self, midi):
        if midi not in self._key_items:
            return
        color = KEY_BLACK if midi % 12 in BLACK_PCS else KEY_WHITE
        for item in self._key_items[midi]:
            try:
                self.itemconfig(item, fill=color)
            except:
                pass


# ══════════════════════════════════════════════════════════════════════
#  波形表示
# ══════════════════════════════════════════════════════════════════════
class WaveformView(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#090909", highlightthickness=0, **kw)
        self._data = None
        self.bind("<Configure>", lambda e: self._redraw())

    def set_data(self, data):
        self._data = data
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return
        self.create_line(0, h // 2, w, h // 2, fill="#1e1e1e", width=1)
        if self._data is None or len(self._data) == 0:
            return
        data = np.nan_to_num(self._data, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(data)
        step = max(1, n // w)
        pts = []
        for i in range(0, n, step):
            pts += [int(i / n * w), int(h / 2 - data[i] * (h / 2 - 4))]
        if len(pts) >= 4:
            self.create_line(*pts, fill="#4a5500", width=4, smooth=False)
            self.create_line(*pts, fill=ACCENT, width=1.5, smooth=False)


# ══════════════════════════════════════════════════════════════════════
#  メインアプリ
# ══════════════════════════════════════════════════════════════════════
class SynthApp:
    def __init__(self, root, model=None, reference=None):
        self.root = root
        self.model = model
        self.reference = reference

        self._params = {"screech": 0.0, "acid": 0.0, "pluck": 0.0}
        self._midi = MIDI_MIN
        self._generating = False

        root.title("CVAE SYNTH")
        root.configure(bg=BG)
        root.resizable(True, True)

        self._build_ui()
        self._update_status()

    # ── UI 構築 ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ヘッダー
        hdr = tk.Frame(root, bg="#111", pady=8)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="CVAE SYNTH",
            bg="#111",
            fg=ACCENT,
            font=("Courier", 22, "bold"),
            padx=20,
        ).pack(side="left")
        tk.Label(
            hdr,
            text="CONDITIONAL VARIATIONAL AUTOENCODER",
            bg="#111",
            fg=DIM,
            font=("Courier", 8),
        ).pack(side="left", padx=8)

        self._led_canvas = tk.Canvas(
            hdr, width=60, height=20, bg="#111", highlightthickness=0
        )
        self._led_canvas.pack(side="right", padx=16)
        self._led_gen = self._led_canvas.create_oval(
            5, 5, 15, 15, fill="#1a2200"
        )
        self._led_play = self._led_canvas.create_oval(
            25, 5, 35, 15, fill="#001a22"
        )
        self._led_canvas.create_text(
            10, 18, text="GEN", fill=DIM, font=("Courier", 6), anchor="n"
        )
        self._led_canvas.create_text(
            30, 18, text="PLY", fill=DIM, font=("Courier", 6), anchor="n"
        )

        tk.Frame(root, bg=ACCENT, height=2).pack(fill="x")

        # メインエリア
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 1))
        right = tk.Frame(main, bg=PANEL2, width=200)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_knobs(left)
        self._build_waveform(left)
        self._build_vector_panel(right)

        # 鍵盤エリア
        kb_frame = tk.Frame(root, bg="#111", pady=10)
        kb_frame.pack(fill="x")

        tk.Label(
            kb_frame,
            text="PITCH SELECT  — クリックで即時生成・再生",
            bg="#111",
            fg=DIM,
            font=("Courier", 8),
            padx=16,
        ).pack(anchor="w")

        self._pitch_label = tk.Label(
            kb_frame,
            text="C2",
            bg="#111",
            fg=ACCENT,
            font=("Courier", 28, "bold"),
            padx=16,
        )
        self._pitch_label.pack(anchor="w")

        self._freq_label = tk.Label(
            kb_frame,
            text="65.4 Hz  |  MIDI 36  |  pitch_n = 0.000",
            bg="#111",
            fg=DIM,
            font=("Courier", 9),
            padx=16,
        )
        self._freq_label.pack(anchor="w")

        kb_scroll = tk.Frame(kb_frame, bg="#111")
        kb_scroll.pack(fill="x", padx=16, pady=(8, 0))

        # ★ on_note に _on_note_and_generate を渡す（押したら即生成）
        self._keyboard = PianoKeyboard(
            kb_scroll, on_note=self._on_note_and_generate
        )
        self._keyboard.pack(side="left")

        # キーボードショートカット
        root.bind("<Left>", lambda e: self._key_shift(-1))
        root.bind("<Right>", lambda e: self._key_shift(+1))
        root.bind("<Down>", lambda e: self._key_shift(-12))
        root.bind("<Up>", lambda e: self._key_shift(+12))
        root.bind("<space>", lambda e: self._generate())

    def _build_knobs(self, parent):
        frame = tk.Frame(parent, bg=PANEL, pady=12)
        frame.pack(fill="x", padx=12)

        tk.Label(
            frame,
            text="CONDITION VECTOR",
            bg=PANEL,
            fg=DIM,
            font=("Courier", 8),
        ).pack(anchor="w")
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(2, 12))

        knob_row = tk.Frame(frame, bg=PANEL)
        knob_row.pack()

        self._knobs = {}
        for name in ("screech", "acid", "pluck"):
            kb = Knob(
                knob_row,
                label=name.upper(),
                color=KNOB_COLORS[name],
                command=lambda v, n=name: self._on_knob(n, v),
            )
            kb.pack(side="left", padx=18)
            self._knobs[name] = kb

        btn_frame = tk.Frame(parent, bg=PANEL)
        btn_frame.pack(fill="x", padx=12, pady=6)
        self._gen_btn = tk.Button(
            btn_frame,
            text="GENERATE  [SPACE]",
            bg=PANEL,
            fg=ACCENT,
            activebackground="#222",
            activeforeground=ACCENT,
            relief="flat",
            font=("Courier", 11, "bold"),
            cursor="hand2",
            bd=0,
            pady=8,
            command=self._generate,
            highlightbackground=ACCENT,
            highlightthickness=1,
        )
        self._gen_btn.pack(fill="x")

    def _build_waveform(self, parent):
        frame = tk.Frame(parent, bg=PANEL, pady=6)
        frame.pack(fill="both", expand=True, padx=12)

        tk.Label(
            frame,
            text="WAVEFORM PREVIEW",
            bg=PANEL,
            fg=DIM,
            font=("Courier", 8),
        ).pack(anchor="w")
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(2, 6))

        self._wave_view = WaveformView(frame, height=90)
        self._wave_view.pack(fill="both", expand=True)

    def _build_vector_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL2, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="COND VECTOR [4D]",
            bg=PANEL2,
            fg=DIM,
            font=("Courier", 8),
        ).pack(anchor="w")
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(2, 12))

        bar_info = [
            ("SCREECH", "screech", ACCENT2),
            ("ACID", "acid", ACID_COL),
            ("PLUCK", "pluck", ACCENT3),
            ("PITCH", "pitch", PITCH_COL),
        ]
        self._bar_cvs = {}
        self._bar_fills = {}
        self._bar_texts = {}

        for label, key, color in bar_info:
            row = tk.Frame(frame, bg=PANEL2)
            row.pack(fill="x", pady=3)
            lf = tk.Frame(row, bg=PANEL2)
            lf.pack(fill="x")
            tk.Label(
                lf,
                text=label,
                bg=PANEL2,
                fg=DIM,
                font=("Courier", 8),
                width=8,
                anchor="w",
            ).pack(side="left")
            tv = tk.Label(
                lf,
                text="0.00",
                bg=PANEL2,
                fg=color,
                font=("Courier", 9, "bold"),
                width=5,
                anchor="e",
            )
            tv.pack(side="right")
            self._bar_texts[key] = tv

            cv = tk.Canvas(row, height=6, bg="#1a1a1a", highlightthickness=0)
            cv.pack(fill="x", pady=(2, 0))
            fill = cv.create_rectangle(0, 0, 0, 6, fill=color, outline="")
            self._bar_cvs[key] = cv
            self._bar_fills[key] = fill

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=8)
        tk.Label(
            frame, text="PYTHON OUTPUT", bg=PANEL2, fg=DIM, font=("Courier", 7)
        ).pack(anchor="w")

        self._vec_text = tk.Text(
            frame,
            height=6,
            width=22,
            bg="#0a0a0a",
            fg=ACID_COL,
            font=("Courier", 8),
            relief="flat",
            bd=1,
            insertbackground=ACCENT,
            state="disabled",
        )
        self._vec_text.pack(fill="x", pady=(4, 0))

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._status_label = tk.Label(
            frame,
            text="READY",
            bg=PANEL2,
            fg=DIM,
            font=("Courier", 8),
            wraplength=160,
            justify="left",
        )
        self._status_label.pack(anchor="w")

    # ── コールバック ──────────────────────────────────────────────────
    def _on_knob(self, name, value):
        self._params[name] = value
        self._update_vector_display()

    def _on_note_and_generate(self, midi):
        """★ 鍵盤を押したとき: ピッチ更新 → 即座に生成・再生"""
        self._midi = midi
        note = midi_to_note_name(midi)
        freq = midi_to_freq(midi)
        pn = pitch_norm(midi)

        self._pitch_label.config(text=note)
        self._freq_label.config(
            text=f"{freq:.1f} Hz  |  MIDI {midi}  |  pitch_n = {pn:.3f}"
        )
        self._update_vector_display()

        # 鍵盤クリックで即生成・再生
        self._generate()

    def _key_shift(self, delta):
        new = max(MIDI_MIN, min(MIDI_MAX, self._midi + delta))
        if new != self._midi:
            self._keyboard.select(new)  # → _on_note_and_generate が呼ばれる

    def _update_vector_display(self):
        pn = pitch_norm(self._midi)
        values = {
            "screech": self._params["screech"],
            "acid": self._params["acid"],
            "pluck": self._params["pluck"],
            "pitch": pn,
        }
        for key, v in values.items():
            cv = self._bar_cvs[key]
            cv.update_idletasks()
            w = cv.winfo_width()
            if w > 0:
                cv.coords(self._bar_fills[key], 0, 0, int(w * v), 6)
            self._bar_texts[key].config(text=f"{v:.2f}")

        s = self._params["screech"]
        a = self._params["acid"]
        p = self._params["pluck"]
        note = midi_to_note_name(self._midi)
        code = (
            f"# Note: {note}  (MIDI {self._midi})\n"
            f"pitch = {self._midi}\n"
            f"cond  = (\n"
            f"  {s:.3f},  # screech\n"
            f"  {a:.3f},  # acid\n"
            f"  {p:.3f},  # pluck\n"
            f")\n"
            f"# cond_vector = [{pn:.3f}, {s:.3f}, {a:.3f}, {p:.3f}]"
        )
        self._vec_text.config(state="normal")
        self._vec_text.delete("1.0", "end")
        self._vec_text.insert("1.0", code)
        self._vec_text.config(state="disabled")

    def _update_status(self):
        model_str = "✓ MODEL LOADED" if self.model else "⚠ NO MODEL (DEMO)"
        sd_str = "✓ AUDIO OK" if HAS_SD else "⚠ NO AUDIO"
        self._status_label.config(
            text=f"{model_str}\n{sd_str}", fg=ACCENT if self.model else DIM
        )

    # ── 生成 ──────────────────────────────────────────────────────────
    def _generate(self):
        if self._generating:
            return
        self._generating = True
        self._gen_btn.config(state="disabled", text="GENERATING...")
        self._led_on(self._led_gen, ACCENT2)
        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        try:
            midi = self._midi
            s = self._params["screech"]
            a = self._params["acid"]
            p = self._params["pluck"]

            waveform = (
                self._run_model(midi, (s, a, p))
                if self.model is not None
                else self._demo_waveform(midi, s, a, p)
            )

            self.root.after(0, lambda: self._post_generate(waveform))
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.root.after(
                0,
                lambda: self._status_label.config(
                    text=f"ERROR:\n{e}", fg=ACCENT2
                ),
            )
            self.root.after(0, self._reset_gen_btn)

    def _run_model(self, midi, cond):
        import tensorflow as tf

        pn = pitch_norm(midi)
        cond_vector = tf.constant([[pn, *cond]], dtype=tf.float32)

        ref = self.reference
        if ref is None:
            t = np.arange(TIME_LENGTH) / SR
            ref = (0.5 * np.sin(2 * np.pi * midi_to_freq(midi) * t))[
                :, None
            ].astype(np.float32)
        else:
            ref = np.array(ref, dtype=np.float32)
            if ref.ndim == 1:
                ref = ref[:, None]

        ref_batch = tf.constant(ref[None, :, :], dtype=tf.float32)
        # encoder は波形のみを受け取る (cond_vector は decoder にだけ渡す)
        z_mean, z_lv = self.model.encoder(ref_batch)
        z = (
            z_mean
            + tf.exp(0.5 * z_lv) * tf.random.normal(tf.shape(z_mean)) * 0.1
        )
        x_hat = tf.squeeze(self.model.decoder([z, cond_vector])).numpy()

        # NaN/Inf チェック
        x_hat = np.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)
        nan_ratio = np.sum(x_hat == 0.0) / len(x_hat)
        if nan_ratio > 0.9:
            print(
                f"[警告] 出力の{nan_ratio*100:.0f}%がNaN/Infでした。重みかリファレンスを確認してください。"
            )
        mx = np.max(np.abs(x_hat))
        if mx > 1e-6:
            x_hat = x_hat / mx * 0.95
        if HAS_SF:
            sf.write("generated.wav", x_hat, SR)
        return x_hat

    def _demo_waveform(self, midi, screech, acid, pluck):
        duration = 1.0
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        phase = 2 * np.pi * midi_to_freq(midi) * t
        s = np.sin(phase)
        for h in range(3, 12, 2):
            s += acid * 0.5 / h * np.sin(phase * h)
        if screech > 0.01:
            s += screech * np.sin(phase * 8 + np.sin(phase * 2) * screech * 4)
            s = np.tanh(s * (1 + screech * 2))
        if pluck > 0.01:
            s *= np.exp(-t * 8 * pluck)
        mx = np.max(np.abs(s))
        if mx > 1e-6:
            s = s / mx * 0.90
        return s.astype(np.float32)

    def _post_generate(self, waveform):
        self._wave_view.set_data(waveform)
        if HAS_SD:
            self._led_on(self._led_play, ACCENT3)
            sd.stop()
            sd.play(waveform, SR)
        self._status_label.config(
            text=f"GENERATED\n{len(waveform)/SR:.2f}s  {SR}Hz", fg=ACCENT
        )
        self._reset_gen_btn()

    def _reset_gen_btn(self):
        self._generating = False
        self._gen_btn.config(state="normal", text="GENERATE  [SPACE]")
        self.root.after(800, lambda: self._led_off(self._led_gen))
        self.root.after(2000, lambda: self._led_off(self._led_play))

    def _led_on(self, item, color):
        self._led_canvas.itemconfig(item, fill=color)

    def _led_off(self, item):
        self._led_canvas.itemconfig(item, fill="#1a2200")


# ══════════════════════════════════════════════════════════════════════
#  エントリーポイント
# ══════════════════════════════════════════════════════════════════════
def main():
    model = None
    reference = None

    if HAS_MODEL:
        try:
            print("[1] モデル読み込み中...")
            import tensorflow as tf

            model = TimeWiseCVAE()
            dummy_x = tf.zeros((1, TIME_LENGTH, 1))
            dummy_cond = tf.zeros((1, 4))
            model((dummy_x, dummy_cond), training=False)
            print(f"    ✓ モデルビルド完了 (params: {model.count_params():,})")

            ckpt = "weights/best_model.weights.h5"
            model.load_weights(ckpt)
            print(f"    ✓ 重みを読み込みました: {ckpt}")

            print("[2] リファレンス音声読み込み中...")
            reference = load_wav("datasets/0013.wav")
            reference = crop_or_pad(reference, TIME_LENGTH)
            print("    ✓ リファレンス読み込み完了")

        except Exception as e:
            import traceback

            print("\n" + "=" * 60)
            print("⚠ モデル読み込み失敗 → デモモードで起動します")
            print("=" * 60)
            traceback.print_exc()
            print("=" * 60 + "\n")
            model = None

    root = tk.Tk()
    SynthApp(root, model=model, reference=reference)
    root.mainloop()


if __name__ == "__main__":
    main()
