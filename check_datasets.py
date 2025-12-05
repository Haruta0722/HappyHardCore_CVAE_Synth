import glob
import numpy as np
from train_spectol import load_wav, wav_to_mel_py
files = glob.glob("datasets/CVAE3/*.wav")
def main():
    for f in files:
        wav = load_wav(f)
        mel = wav_to_mel_py(wav)

        if mel.size == 0 or mel.shape[0] == 0:
            print("EMPTY MEL!! →", f)

        if np.isnan(mel).any() or np.isinf(mel).any():
            print("NaN/Inf DETECTED →", f)

        if wav is None:
            print("LOAD FAILED →", f)

if __name__ == "__main__":
    main()      