import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

print("imports ok, TF version:", tf.__version__)
