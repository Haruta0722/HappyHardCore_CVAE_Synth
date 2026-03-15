"""
gui.py  ―  CVAE Synth GUI (DDSPパラメータ個別操作版)

【操作フロー】
  1. ノブで音色ブレンド比率を設定
  2. 鍵盤でピッチを選択 → 自動生成・再生
  3. VAEが推論した DDSPParams を右パネルで確認
  4. 各スライダーでパラメータを手動上書き → RE-SYNTH ボタンで再合成

【パネル構成】
  左: 音色ブレンドノブ / GENERATE ボタン / 波形表示
  右上: DDSP パラメータスライダー (Oscillator / ADSR / Filter / Noise)
  右下: ブレンドベクトル表示 / ステータス
"""

import tkinter as tk
from tkinter import ttk
import math
import threading
import numpy as np

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

try:
    import tensorflow as tf
    from cvae import TimeWiseCVAE
    from dsp import DDSPParams, synthesize_numpy
    from config import (
        TIME_LENGTH,
        TIMBRE_VOCAB,
        TIMBRE_EMBED_DIM,
        NUM_HARMONICS,
    )

    HAS_MODEL = True
except Exception as e:
    HAS_MODEL = False
    TIME_LENGTH = 62400
    TIMBRE_VOCAB = 3
    NUM_HARMONICS = 32
    print(f"[警告] モデル読み込みスキップ: {e}")

    # ダミークラス
    class DDSPParams:
        def __init__(self, **kw):
            pass


# ══════════════════════════════════════════════════════════════
#  定数 / カラー
# ══════════════════════════════════════════════════════════════
SR = 48000
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHITE_PCS = [0, 2, 4, 5, 7, 9, 11]
BLACK_PCS = {1, 3, 6, 8, 10}
MIDI_MIN, MIDI_MAX = 36, 71

BG = "#0d0d0d"
PANEL = "#161616"
PANEL2 = "#1a1a1a"
PANEL3 = "#111111"
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
TIMBRE_NAMES = ["screech", "acid", "pluck"]

# DDSPパラメータスライダーの定義
# (キー, 表示ラベル, 色, グループ)
DDSP_SLIDERS = [
    # Oscillator
    (
        "harm_brightness",
        "BRIGHTNESS",
        ACCENT,
        "OSC",
    ),  # 高倍音の強さ (harmonic_amps から計算)
    # ADSR
    ("attack", "ATTACK", ACCENT3, "ENV"),
    ("decay", "DECAY", ACCENT3, "ENV"),
    ("sustain", "SUSTAIN", ACCENT3, "ENV"),
    ("release", "RELEASE", ACCENT3, "ENV"),
    # Filter
    ("cutoff", "CUTOFF", ACID_COL, "FLT"),
    ("resonance", "RESONANCE", ACID_COL, "FLT"),
    # Noise
    ("noise_amount", "NOISE", ACCENT2, "NOI"),
]


# ══════════════════════════════════════════════════════════════
#  ヘルパー
# ══════════════════════════════════════════════════════════════
def midi_to_note_name(m):
    return NOTE_NAMES[m % 12] + str(m // 12 - 1)


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pitch_norm(m):
    return (m - 36) / 35.0


def weights_to_tensor(weights):
    vals = np.array([weights[n] for n in TIMBRE_NAMES], dtype=np.float32)
    total = vals.sum()
    vals = (
        vals / total
        if total > 1e-6
        else np.ones(TIMBRE_VOCAB, dtype=np.float32) / TIMBRE_VOCAB
    )
    return tf.constant(vals[None, :], dtype=tf.float32)


def brightness_from_amps(harmonic_amps):
    """harmonic_amps → 高倍音の重み平均 (0〜1) をブライトネスとして返す"""
    amps = np.array(harmonic_amps, dtype=np.float32)
    n = len(amps)
    if n == 0 or amps.sum() < 1e-8:
        return 0.5
    weights = np.linspace(0.0, 1.0, n)
    return float(np.dot(amps / (amps.sum() + 1e-8), weights))


def brightness_to_amps(brightness, n=NUM_HARMONICS):
    """ブライトネス値 → softmax的なharmonic_amps"""
    weights = np.linspace(0.0, 1.0, n)
    # brightがゼロなら基音のみ、1なら高次倍音が強い
    logits = (weights - 0.5) * brightness * 10.0
    amps = np.exp(logits - logits.max())
    return (amps / amps.sum()).tolist()


# ══════════════════════════════════════════════════════════════
#  ノブウィジェット
# ══════════════════════════════════════════════════════════════
class Knob(tk.Canvas):
    START_DEG = 225
    SWEEP = 270

    def __init__(self, parent, label, color, command=None, size=68, **kw):
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

    def set_silent(self, v):
        """コールバックなしで値を設定"""
        self._value = max(0.0, min(1.0, v))
        self._draw()

    def _draw(self):
        self.delete("all")
        s = self.size
        cx = cy = s // 2
        ro = s // 2 - 4
        rt = ro - 7
        ri = ro - 12
        self.create_oval(
            cx - ro,
            cy - ro,
            cx + ro,
            cy + ro,
            fill="#111",
            outline=BORDER,
            width=1.5,
        )
        self.create_oval(
            cx - rt - 4,
            cy - rt - 4,
            cx + rt + 4,
            cy + rt + 4,
            fill="#1e1e1e",
            outline="#2e2e2e",
            width=1,
        )
        self._arc(cx, cy, rt, 0, 1.0, "#2a2a2a", width=4)
        if self._value > 0.005:
            self._arc(cx, cy, rt, 0, self._value, self.color, width=4)
        a = math.radians(self.START_DEG + self.SWEEP * self._value - 90)
        self.create_line(
            cx,
            cy,
            cx + ri * math.cos(a),
            cy + ri * math.sin(a),
            fill=self.color,
            width=2.5,
            capstyle=tk.ROUND,
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
        d = getattr(e, "delta", 0)
        if e.num == 4:
            d = 120
        if e.num == 5:
            d = -120
        self.value = self._value + d / 12000


# ══════════════════════════════════════════════════════════════
#  ピアノ鍵盤
# ══════════════════════════════════════════════════════════════
class PianoKeyboard(tk.Canvas):
    WW, WH = 30, 90
    BW, BH = 19, 54

    def __init__(self, parent, on_note, **kw):
        self._white_keys = [
            m for m in range(MIDI_MIN, MIDI_MAX + 1) if m % 12 in WHITE_PCS
        ]
        super().__init__(
            parent,
            width=len(self._white_keys) * self.WW + 2,
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
            ids = [
                self.create_rectangle(
                    x,
                    1,
                    x + self.WW - 1,
                    self.WH,
                    fill=KEY_WHITE,
                    outline="#888",
                    width=1,
                    tags=f"key{m}",
                )
            ]
            if m % 12 == 0:
                ids.append(
                    self.create_text(
                        x + self.WW // 2,
                        self.WH - 10,
                        text=midi_to_note_name(m),
                        font=("Courier", 6),
                        fill="#666",
                        tags=f"key{m}",
                    )
                )
            self._key_items[m] = ids
        for m in range(MIDI_MIN, MIDI_MAX + 1):
            if m % 12 not in BLACK_PCS:
                continue
            left = m - 1
            while left % 12 not in WHITE_PCS:
                left -= 1
            x = white_x[left] + self.WW - self.BW // 2
            self._key_items[m] = [
                self.create_rectangle(
                    x,
                    0,
                    x + self.BW,
                    self.BH,
                    fill=KEY_BLACK,
                    outline="#000",
                    width=1,
                    tags=f"key{m}",
                )
            ]
        self._highlight(self._active)

    def _click(self, e):
        items = self.find_overlapping(e.x - 1, e.y - 1, e.x + 1, e.y + 1)
        hit = None
        for item in reversed(items):
            for tag in self.gettags(item):
                if tag.startswith("key"):
                    try:
                        m = int(tag[3:])
                        if m in self._key_items:
                            hit = m
                            break
                    except ValueError:
                        pass
            if hit is not None:
                break
        if hit is not None:
            self.select(hit)

    def select(self, midi):
        old = self._active
        self._active = midi
        self._unhighlight(old)
        self._highlight(midi)
        self.on_note(midi)

    def _highlight(self, m):
        if m not in self._key_items:
            return
        for item in self._key_items[m]:
            try:
                self.itemconfig(item, fill=KEY_ACT)
            except:
                pass

    def _unhighlight(self, m):
        if m not in self._key_items:
            return
        c = KEY_BLACK if m % 12 in BLACK_PCS else KEY_WHITE
        for item in self._key_items[m]:
            try:
                self.itemconfig(item, fill=c)
            except:
                pass


# ══════════════════════════════════════════════════════════════
#  波形表示
# ══════════════════════════════════════════════════════════════
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
        data = np.nan_to_num(self._data)
        n = len(data)
        step = max(1, n // w)
        pts = []
        for i in range(0, n, step):
            pts += [int(i / n * w), int(h / 2 - data[i] * (h / 2 - 4))]
        if len(pts) >= 4:
            self.create_line(*pts, fill="#4a5500", width=4, smooth=False)
            self.create_line(*pts, fill=ACCENT, width=1.5, smooth=False)


# ══════════════════════════════════════════════════════════════
#  メインアプリ
# ══════════════════════════════════════════════════════════════
class SynthApp:
    def __init__(self, root, model=None):
        self.root = root
        self.model = model
        self._params = {"screech": 1.0, "acid": 0.0, "pluck": 0.0}
        self._midi = MIDI_MIN
        self._generating = False
        # 現在のDDSPParams (モデル推論値、GUIで上書き可能)
        self._ddsp = DDSPParams() if HAS_MODEL else None
        # GUIスライダーの値 (キー → DoubleVar)
        self._slider_vars = {}

        root.title("CVAE SYNTH  —  DDSP Parameter Control")
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
            text="DDSP PARAMETER CONTROL",
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

        # メイン (左 + 右)
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg=PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(0, 1))
        right = tk.Frame(main, bg=PANEL2, width=260)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # ← 右パネルを先にビルド (_bar_cvs 等を先に作る)
        self._build_ddsp_panel(right)
        self._build_knobs(left)
        self._build_waveform(left)
        self._build_keyboard(root)

        root.bind("<Left>", lambda e: self._key_shift(-1))
        root.bind("<Right>", lambda e: self._key_shift(+1))
        root.bind("<Down>", lambda e: self._key_shift(-12))
        root.bind("<Up>", lambda e: self._key_shift(+12))
        root.bind("<space>", lambda e: self._generate())

    def _build_knobs(self, parent):
        frame = tk.Frame(parent, bg=PANEL, pady=12)
        frame.pack(fill="x", padx=12)
        tk.Label(
            frame, text="TIMBRE BLEND", bg=PANEL, fg=DIM, font=("Courier", 8)
        ).pack(anchor="w")
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", pady=(2, 12))
        row = tk.Frame(frame, bg=PANEL)
        row.pack()
        self._knobs = {}
        for name in TIMBRE_NAMES:
            kb = Knob(
                row,
                label=name.upper(),
                color=KNOB_COLORS[name],
                command=lambda v, n=name: self._on_knob(n, v),
            )
            kb.pack(side="left", padx=14)
            self._knobs[name] = kb
        self._knobs["screech"].set_silent(1.0)
        self._params["screech"] = 1.0

        btn_frame = tk.Frame(parent, bg=PANEL)
        btn_frame.pack(fill="x", padx=12, pady=4)
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

    def _build_keyboard(self, root):
        kb_frame = tk.Frame(root, bg="#111", pady=10)
        kb_frame.pack(fill="x")
        tk.Label(
            kb_frame,
            text="PITCH SELECT — クリックで即時生成・再生",
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
        self._keyboard = PianoKeyboard(
            kb_scroll, on_note=self._on_note_and_generate
        )
        self._keyboard.pack(side="left")

    def _build_ddsp_panel(self, parent):
        """右パネル: DDSPパラメータスライダー + ブレンド表示"""
        canvas = tk.Canvas(parent, bg=PANEL2, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            parent, orient="vertical", command=canvas.yview
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=PANEL2)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        # ── DDSPパラメータスライダー ────────────────────────────────
        tk.Label(
            frame,
            text="DDSP PARAMETERS",
            bg=PANEL2,
            fg=ACCENT,
            font=("Courier", 9, "bold"),
            padx=8,
            pady=6,
        ).pack(anchor="w")
        tk.Frame(frame, bg=ACCENT, height=1).pack(fill="x", padx=8)

        current_group = [None]
        for key, label, color, group in DDSP_SLIDERS:
            if group != current_group[0]:
                current_group[0] = group
                gframe = tk.Frame(frame, bg=PANEL3, padx=6, pady=4)
                gframe.pack(fill="x", padx=8, pady=(6, 0))
                group_labels = {
                    "OSC": "▶ OSCILLATOR",
                    "ENV": "▶ ENVELOPE (ADSR)",
                    "FLT": "▶ FILTER",
                    "NOI": "▶ NOISE",
                }
                tk.Label(
                    gframe,
                    text=group_labels.get(group, group),
                    bg=PANEL3,
                    fg=DIM,
                    font=("Courier", 7),
                ).pack(anchor="w")

            row = tk.Frame(frame, bg=PANEL2, padx=8)
            row.pack(fill="x", pady=2)
            tk.Label(
                row,
                text=f"{label:<12}",
                bg=PANEL2,
                fg=color,
                font=("Courier", 8),
                width=12,
                anchor="w",
            ).pack(side="left")

            var = tk.DoubleVar(value=0.5)
            self._slider_vars[key] = var

            sl = tk.Scale(
                row,
                variable=var,
                from_=0.0,
                to=1.0,
                resolution=0.01,
                orient="horizontal",
                length=120,
                bg=PANEL2,
                fg=color,
                troughcolor="#222",
                highlightthickness=0,
                activebackground=color,
                showvalue=False,
                command=lambda v, k=key: self._on_ddsp_slider(k, float(v)),
            )
            sl.pack(side="left", padx=4)

            val_label = tk.Label(
                row,
                text="0.50",
                bg=PANEL2,
                fg=color,
                font=("Courier", 8),
                width=4,
            )
            val_label.pack(side="left")
            var.trace_add(
                "write",
                lambda *a, vl=val_label, vr=var: vl.config(
                    text=f"{vr.get():.2f}"
                ),
            )

        # RE-SYNTH ボタン
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Button(
            frame,
            text="RE-SYNTH (手動パラメータで再合成)",
            bg=PANEL3,
            fg=ACID_COL,
            activebackground="#222",
            activeforeground=ACID_COL,
            relief="flat",
            font=("Courier", 8, "bold"),
            cursor="hand2",
            bd=0,
            pady=6,
            command=self._resynth,
            highlightbackground=ACID_COL,
            highlightthickness=1,
        ).pack(fill="x", padx=8)

        # ── ブレンドバー ────────────────────────────────────────────
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(
            frame,
            text="TIMBRE BLEND",
            bg=PANEL2,
            fg=DIM,
            font=("Courier", 8),
            padx=8,
        ).pack(anchor="w")
        self._bar_cvs = {}
        self._bar_fills = {}
        self._bar_texts = {}
        for name, color in zip(TIMBRE_NAMES, [ACCENT2, ACID_COL, ACCENT3]):
            r = tk.Frame(frame, bg=PANEL2, padx=8)
            r.pack(fill="x", pady=2)
            tk.Label(
                r,
                text=name.upper(),
                bg=PANEL2,
                fg=DIM,
                font=("Courier", 7),
                width=8,
                anchor="w",
            ).pack(side="left")
            tv = tk.Label(
                r,
                text="0.00",
                bg=PANEL2,
                fg=color,
                font=("Courier", 8, "bold"),
                width=4,
                anchor="e",
            )
            tv.pack(side="right")
            cv = tk.Canvas(r, height=5, bg="#1a1a1a", highlightthickness=0)
            cv.pack(fill="x", pady=(2, 0))
            fill = cv.create_rectangle(0, 0, 0, 5, fill=color, outline="")
            self._bar_cvs[name] = cv
            self._bar_fills[name] = fill
            self._bar_texts[name] = tv

        # ステータス
        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)
        self._status_label = tk.Label(
            frame,
            text="READY",
            bg=PANEL2,
            fg=DIM,
            font=("Courier", 8),
            wraplength=220,
            justify="left",
            padx=8,
        )
        self._status_label.pack(anchor="w")

    # ── コールバック ──────────────────────────────────────────────────
    def _on_knob(self, name, value):
        self._params[name] = value
        self._update_blend_bars()

    def _on_ddsp_slider(self, key, value):
        """スライダーが動いたとき DDSPParams を更新 (RE-SYNTHまで音は変わらない)"""
        if self._ddsp is None:
            return
        if key == "harm_brightness":
            self._ddsp.harmonic_amps = brightness_to_amps(value)
        else:
            setattr(self._ddsp, key, value)

    def _on_note_and_generate(self, midi):
        self._midi = midi
        self._pitch_label.config(text=midi_to_note_name(midi))
        self._freq_label.config(
            text=f"{midi_to_freq(midi):.1f} Hz  |  MIDI {midi}  |  pitch_n = {pitch_norm(midi):.3f}"
        )
        self._update_blend_bars()
        self._generate()

    def _key_shift(self, delta):
        new = max(MIDI_MIN, min(MIDI_MAX, self._midi + delta))
        if new != self._midi:
            self._keyboard.select(new)

    def _update_blend_bars(self):
        if not hasattr(self, "_bar_cvs"):
            return
        raw = np.array(
            [self._params[n] for n in TIMBRE_NAMES], dtype=np.float32
        )
        total = raw.sum()
        norm = raw / total if total > 1e-6 else np.ones(3, dtype=np.float32) / 3
        for i, name in enumerate(TIMBRE_NAMES):
            cv = self._bar_cvs[name]
            cv.update_idletasks()
            w = cv.winfo_width()
            if w > 0:
                cv.coords(self._bar_fills[name], 0, 0, int(w * norm[i]), 5)
            self._bar_texts[name].config(text=f"{norm[i]:.2f}")

    def _update_ddsp_sliders(self, ddsp: "DDSPParams"):
        """DDSPParams の値をスライダーに反映"""
        if not HAS_MODEL:
            return
        mapping = {
            "harm_brightness": brightness_from_amps(ddsp.harmonic_amps),
            "attack": ddsp.attack,
            "decay": ddsp.decay,
            "sustain": ddsp.sustain,
            "release": ddsp.release,
            "cutoff": ddsp.cutoff,
            "resonance": ddsp.resonance,
            "noise_amount": ddsp.noise_amount,
        }
        for key, val in mapping.items():
            if key in self._slider_vars:
                self._slider_vars[key].set(float(np.clip(val, 0.0, 1.0)))

    def _update_status(self):
        s = "✓ MODEL LOADED" if self.model else "⚠ NO MODEL (DEMO)"
        a = "✓ AUDIO OK" if HAS_SD else "⚠ NO AUDIO"
        self._status_label.config(
            text=f"{s}\n{a}", fg=ACCENT if self.model else DIM
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
            if self.model is not None:
                waveform, ddsp = self._run_model()
                self.root.after(0, lambda: self._post_generate(waveform, ddsp))
            else:
                waveform = self._demo_waveform()
                self.root.after(0, lambda: self._post_generate(waveform, None))
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

    def _run_model(self):
        """
        VAEで DDSPParams を推論 → dsp.synthesize_numpy() で音声合成。
        推論したパラメータをスライダーに反映し、GUIで上書き可能にする。
        """
        pn = np.float32(pitch_norm(self._midi))
        pitch_t = tf.constant([pn], dtype=tf.float32)
        timbre_w = weights_to_tensor(self._params)

        # VAEでパラメータ推論
        ddsp = self.model.infer_ddsp_params(pitch_t, timbre_weights=timbre_w)
        self._ddsp = ddsp

        # NumPy DSPで合成 (TF不要・マイコンと同じパス)
        waveform = synthesize_numpy(
            ddsp, sr=SR, time_length=TIME_LENGTH, fast_filter=True
        )
        waveform = np.nan_to_num(waveform)
        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = waveform / peak * 0.95

        if HAS_SF:
            sf.write("generated.wav", waveform, SR)
        return waveform.astype(np.float32), ddsp

    def _resynth(self):
        """スライダーで手動設定したパラメータで再合成"""
        if self._ddsp is None:
            self._status_label.config(
                text="先にGENERATEしてください", fg=ACCENT2
            )
            return
        self._ddsp.f0_hz = float(midi_to_freq(self._midi))
        waveform = synthesize_numpy(
            self._ddsp, sr=SR, time_length=TIME_LENGTH, fast_filter=True
        )
        waveform = np.nan_to_num(waveform)
        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = waveform / peak * 0.95
        if HAS_SF:
            sf.write("resynth.wav", waveform.astype(np.float32), SR)
        self._wave_view.set_data(waveform)
        if HAS_SD:
            sd.stop()
            sd.play(waveform.astype(np.float32), SR)
        self._status_label.config(text="RE-SYNTH 完了", fg=ACID_COL)

    def _demo_waveform(self):
        midi = self._midi
        raw = np.array(
            [self._params[n] for n in TIMBRE_NAMES], dtype=np.float32
        )
        total = raw.sum()
        norm = raw / total if total > 1e-6 else np.ones(3) / 3
        screech, acid, pluck = norm
        t = np.linspace(0, TIME_LENGTH / SR, TIME_LENGTH, endpoint=False)
        phase = 2 * np.pi * midi_to_freq(midi) * t
        s = np.sin(phase)
        for h in range(3, 12, 2):
            s += acid * 0.5 / h * np.sin(phase * h)
        if screech > 0.01:
            s += screech * np.sin(phase * 8 + np.sin(phase * 2) * screech * 4)
            s = np.tanh(s * (1 + screech * 2))
        if pluck > 0.01:
            s *= np.exp(-t * 8 * pluck)
        peak = np.max(np.abs(s))
        if peak > 1e-6:
            s = s / peak * 0.90
        return s.astype(np.float32)

    def _post_generate(self, waveform, ddsp):
        self._wave_view.set_data(waveform)
        if ddsp is not None:
            self._update_ddsp_sliders(ddsp)
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


# ══════════════════════════════════════════════════════════════
#  エントリーポイント
# ══════════════════════════════════════════════════════════════
def main():
    model = None
    if HAS_MODEL:
        try:
            import tensorflow as tf

            print("[1] モデルを構築中...")
            model = TimeWiseCVAE()
            dummy_audio = tf.zeros([1, TIME_LENGTH, 1], dtype=tf.float32)
            dummy_pitch = tf.zeros([1], dtype=tf.float32)
            dummy_timbre = tf.zeros([1], dtype=tf.int32)
            model((dummy_audio, dummy_pitch, dummy_timbre), training=False)
            print(f"    ✓ モデルビルド完了 (params: {model.count_params():,})")
            import os

            ckpt = "checkpoints/best_weights.weights.h5"
            if os.path.exists(ckpt):
                model.load_weights(ckpt)
                print(f"    ✓ 重みを読み込みました: {ckpt}")
            else:
                print(f"    ⚠ 重みファイルなし → デモモードで起動")
                model = None
        except Exception as e:
            import traceback

            print("⚠ モデル読み込み失敗 → デモモードで起動")
            traceback.print_exc()
            model = None

    root = tk.Tk()
    SynthApp(root, model=model)
    root.mainloop()


if __name__ == "__main__":
    main()
