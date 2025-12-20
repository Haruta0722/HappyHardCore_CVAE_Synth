import os
import librosa
import soundfile as sf

TARGET_SR = 16000

root_dir = "datasets"   # ここに一番上のフォルダを指定
out_root = "datasets_2"  # 変換後を保存するフォルダ

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            in_path = os.path.join(root, file)

            # 元のフォルダ構造を保ったまま出力
            rel_path = os.path.relpath(in_path, root_dir)
            out_path = os.path.join(out_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 読み込み & リサンプリング
            wav, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

            # 書き込み
            sf.write(out_path, wav, TARGET_SR)

            print(f"converted: {in_path} -> {out_path}")