from model import build_decoder, build_encoder, TimeWiseCVAE
from create_datasets import make_dataset_from_synth_csv
import tensorflow as tf


def train_model():
    dataset = make_dataset_from_synth_csv("dataset.csv", batch_size=16)
    model = TimeWiseCVAE()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    model.fit(dataset, epochs=100)  # (waveform, cond)
