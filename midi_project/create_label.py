import csv
from pathlib import Path

# ===== 設定 =====
BASE_DIR = Path("datasets")
OUT_CSV = "dataset.csv"

# C2(36) 〜 B4(71)
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

note_to_pitch = {}
pitch = 36
for octave in range(2, 5):  # 2,3,4
    for note in notes:
        name = f"{note}{octave}"
        note_to_pitch[name] = pitch
        pitch += 1
        if pitch > 71:
            break

# ===== CSV 作成 =====
rows = []

for folder, pitch_value in note_to_pitch.items():
    folder_path = BASE_DIR / folder

    for i in range(1, 37):  # 0001.wav ～ 0039.wav
        fname = f"{i:04d}.wav"
        wav_path = folder_path / fname

        screech = acid = pluck = 0
        if 1 <= i <= 11:
            screech = 1
        elif 12 <= i <= 23:
            acid = 1
        elif 24 <= i <= 36:
            pluck = 1

        rows.append(
            {
                "path": str(wav_path),
                "pitch": pitch_value,
                "screech": screech,
                "acid": acid,
                "pluck": pluck,
            }
        )

# ===== 書き込み =====
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["path", "pitch", "screech", "acid", "pluck"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"{OUT_CSV} を作成しました（{len(rows)} 行）")
