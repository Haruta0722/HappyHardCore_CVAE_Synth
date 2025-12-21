
import pandas as pd

# 削除対象のパス一覧
remove_paths = [
    "datasets/A4/0012.wav",
    "datasets/A4/0020.wav",
    "datasets/A3/0012.wav",
    "datasets/A3/0020.wav",
    "datasets/B2/0013.wav",
    "datasets/B2/0020.wav",
    "datasets/D#3/0013.wav",
    "datasets/D#3/0020.wav",
    "datasets/D#4/0012.wav",
    "datasets/D#4/0020.wav",
    "datasets/B3/0012.wav",
    "datasets/B3/0021.wav",
    "datasets/B4/0012.wav",
    "datasets/B4/0020.wav",
    "datasets/A2/0013.wav",
    "datasets/A2/0020.wav",
    "datasets/D#2/0013.wav",
    "datasets/D#2/0020.wav",
    "datasets/E2/0013.wav",
    "datasets/E2/0020.wav",
    "datasets/F4/0012.wav",
    "datasets/F4/0020.wav",
    "datasets/F3/0013.wav",
    "datasets/F3/0020.wav",
    "datasets/F2/0013.wav",
    "datasets/F2/0020.wav",
    "datasets/E3/0013.wav",
    "datasets/E3/0020.wav",
    "datasets/E4/0012.wav",
    "datasets/E4/0020.wav",
    "datasets/G#3/0012.wav",
    "datasets/G#3/0020.wav",
    "datasets/G#4/0012.wav",
    "datasets/G#4/0020.wav",
    "datasets/C#2/0013.wav",
    "datasets/C#2/0020.wav",
    "datasets/G#2/0013.wav",
    "datasets/G#2/0020.wav",
    "datasets/C#3/0013.wav",
    "datasets/C#3/0020.wav",
    "datasets/C#4/0012.wav",
    "datasets/C#4/0020.wav",
    "datasets/F#4/0012.wav",
    "datasets/F#4/0020.wav",
    "datasets/G2/0013.wav",
    "datasets/G2/0020.wav",
    "datasets/F#3/0013.wav",
    "datasets/F#3/0020.wav",
    "datasets/D3/0013.wav",
    "datasets/D3/0020.wav",
    "datasets/D4/0012.wav",
    "datasets/D4/0020.wav",
    "datasets/D2/0013.wav",
    "datasets/D2/0020.wav",
    "datasets/F#2/0013.wav",
    "datasets/F#2/0020.wav",
    "datasets/G4/0012.wav",
    "datasets/G4/0020.wav",
    "datasets/G3/0013.wav",
    "datasets/G3/0020.wav",
    "datasets/C3/0013.wav",
    "datasets/C3/0020.wav",
    "datasets/C4/0012.wav",
    "datasets/C4/0020.wav",
    "datasets/C2/0013.wav",
    "datasets/C2/0017.wav",
    "datasets/C2/0019.wav",
    "datasets/C2/0018.wav",
    "datasets/C2/0020.wav",
    "datasets/C2/0021.wav",
    "datasets/A#2/0013.wav",
    "datasets/A#2/0020.wav",
    "datasets/A#4/0012.wav",
    "datasets/A#4/0020.wav",
    "datasets/A#3/0012.wav",
    "datasets/A#3/0020.wav",
]

# CSV読み込み
df = pd.read_csv("dataset.csv")

# 削除前の行数
before = len(df)

# path が remove_paths に含まれる行を削除
df = df[~df["path"].isin(remove_paths)]

# 削除後の行数
after = len(df)

print(f"削除された行数: {before - after}")

# 新しいCSVとして保存（上書きしたくない場合は名前を変える）
df.to_csv("dataset_filtered.csv", index=False)
