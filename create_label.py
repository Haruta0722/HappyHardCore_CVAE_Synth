import os
import csv
import numpy as np

base_dir = "datasets"
input_dir = os.path.join(base_dir, "input_data")

# CVAEごとのパラメータ設定例（線形に増やすだけでもOK）
num_dirs = 12
attack = [0.0, 0.5, 1.0]
distortion = [0.0, 0.5, 1.0]
thickness = [0.0, 0.5, 1.0]
center_tone = [0.0, 0.5, 1.0]

with open(os.path.join(base_dir, "labels.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "input_path",
            "output_path",
            "attack",
            "distortion",
            "thickness",
            "center_tone",
        ]
    )

    for i in range(1, num_dirs + 1):
        cvae_dir = f"CVAE{i}"
        input_path = os.path.join(input_dir, f"{i:04d}.wav")
        i: int = 1
        for a in attack:
            for d in distortion:
                for t in thickness:
                    for c in center_tone:

                        output_path = os.path.join(
                            base_dir, cvae_dir, f"{i:04d}.wav"
                        )
                        writer.writerow(
                            [
                                input_path,
                                output_path,
                                a,
                                d,
                                t,
                                c,
                            ]
                        )
                        i += 1
