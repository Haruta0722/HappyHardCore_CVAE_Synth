import tensorflow as tf

STFT_FFTS = [256, 512, 1024]  # 頭打ちになるたびに2048,4096を追加
MEL_BINS = 80
SR = 48000
# データ CSV パス
LABEL_CSV = "datasets/labels.csv"
BASE_DIR = "."  # CSV内の相対パス基準（必要なら変更）


def get_mel_matrix(n_fft, n_mels=MEL_BINS, sr=SR, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )


def spectral_convergence(S_mag, S_hat_mag, eps=1e-7):
    # S, S_hat: shape [B, frames, freq]

    diff = S_mag - S_hat_mag  # [B, frames, freq]

    # per-example Frobenius norm: sqrt(sum(square(x), axes=(1,2)))
    num_per_example = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[1, 2]))
    den_per_example = tf.sqrt(tf.reduce_sum(tf.square(S_mag), axis=[1, 2]))

    sc_per_example = num_per_example / (den_per_example + eps)  # [B]
    return tf.reduce_mean(sc_per_example)  # scalar (平均)


def magnitude_l1(S_mag, S_hat_mag):

    return tf.reduce_mean(tf.abs(S_mag - S_hat_mag))


def log_mag_l1(S_mag, S_hat_mag):

    return tf.reduce_mean(tf.abs(tf.math.log(S_mag) - tf.math.log(S_hat_mag)))


def mel_l1(S_mag, S_hat_mag, n_fft=1024, hop=256, n_mels=MEL_BINS, eps=1e-7):

    # mel 行列は固定なので勾配を stop！
    mel_mat = tf.stop_gradient(
        tf.cast(get_mel_matrix(n_fft, n_mels=n_mels), S_mag.dtype)
    )

    mel = tf.matmul(S_mag, mel_mat)
    mel_hat = tf.matmul(S_hat_mag, mel_mat)

    mel = tf.maximum(mel, eps)
    mel_hat = tf.maximum(mel_hat, eps)

    return tf.reduce_mean(tf.abs(tf.math.log(mel) - tf.math.log(mel_hat)))


def mel_kl(S_mag, S_hat_mag, n_fft=1024, hop=256, n_mels=MEL_BINS, eps=1e-7):

    # mel 行列は固定
    mel_mat = tf.stop_gradient(
        tf.cast(get_mel_matrix(n_fft, n_mels=n_mels), S_mag.dtype)
    )

    mel = tf.matmul(S_mag, mel_mat)
    mel_hat = tf.matmul(S_hat_mag, mel_mat)

    mel = tf.maximum(mel, eps)
    mel_hat = tf.maximum(mel_hat, eps)

    # フレームごとに正規化（分布化）
    mel_sum = tf.reduce_sum(mel, axis=-1, keepdims=True)
    mel_hat_sum = tf.reduce_sum(mel_hat, axis=-1, keepdims=True)

    P = mel / mel_sum
    Q = mel_hat / mel_hat_sum

    # KL(P || Q)
    kl = tf.reduce_sum(
        P * (tf.math.log(P + eps) - tf.math.log(Q + eps)), axis=-1
    )

    return tf.reduce_mean(kl)


def temporal_diff_l1(S_mag, S_hat_mag):
    dy = S_mag[:, 1:] - S_mag[:, :-1]
    dy_hat = S_hat_mag[:, 1:] - S_hat_mag[:, :-1]
    return tf.reduce_mean(tf.abs(dy - dy_hat))


def STFT_loss(S_mag, S_hat_mag, fft_size, hop_size):

    diff = magnitude_l1(S_mag, S_hat_mag)
    log_loss = log_mag_l1(S_mag, S_hat_mag)
    sc = spectral_convergence(S_mag, S_hat_mag)

    stft_loss = diff + log_loss + sc

    return stft_loss


def Loss(y, y_hat, fft_size, hop_size, eps=1e-7):

    S = tf.stop_gradient(
        tf.signal.stft(
            y, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size
        )
    )
    S_hat = tf.signal.stft(
        y_hat, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size
    )

    S_mag = tf.maximum(tf.abs(S), eps)
    S_hat_mag = tf.maximum(tf.abs(S_hat), eps)

    stft_loss = STFT_loss(S_mag, S_hat_mag, fft_size, hop_size)

    mel_loss = mel_l1(S_mag, S_hat_mag, n_fft=fft_size, hop=hop_size)

    diff_loss = temporal_diff_l1(S_mag, S_hat_mag)

    return stft_loss, mel_loss, diff_loss


def Loss_for_test(y, y_hat, fft_size, hop_size, eps=1e-7):

    S = tf.stop_gradient(
        tf.signal.stft(
            y, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size
        )
    )
    S_hat = tf.signal.stft(
        y_hat, frame_length=fft_size, frame_step=hop_size, fft_length=fft_size
    )

    S_mag = tf.maximum(tf.abs(S), eps)
    S_hat_mag = tf.maximum(tf.abs(S_hat), eps)

    mel_loss = mel_kl(S_mag, S_hat_mag, n_fft=fft_size, hop=hop_size)

    return mel_loss
