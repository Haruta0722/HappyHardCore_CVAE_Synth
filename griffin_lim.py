# pip install librosa soundfile
import numpy as np
import librosa
import soundfile as sf

# --- 入力例: mel_spec が与えられている場合 ---
# mel_spec: shape (n_mels, t) — これは "パワー"（＝振幅^power）か "振幅" か、
# または dB 表現かで処理が変わります。下は一般的な「パワー」(power=2.0) のケース。

sr = 32000
n_fft = 2048
hop_length = 256
win_length = None  # None なら n_fft を使う
n_mels = 80
power = 2.0  # melスペクトルが振幅スペクトルなら 1.0, パワーなら 2.0

# mel_spec を用意（例：モデルの出力など）。ここではダミー：
# mel_spec = np.load('mel.npy')  # shape (n_mels, T)
# もし log-mel（dB）なら librosa.db_to_power を使って戻す必要あり:
# mel_power = librosa.db_to_power(mel_db)

# ランダムのダミー（実際は上で用意した mel_spec を使う）
# mel_power = np.abs(np.random.randn(n_mels, 200)).astype(np.float32)

def mel_to_wave_via_griffinlim(mel_power, sr=sr, n_fft=n_fft,
                               hop_length=hop_length, win_length=win_length,
                               n_iter=60, power=power, n_mels=n_mels):
    """
    mel_power: メルスペクトログラム（非対数、パワーまたは振幅）
    n_iter: Griffin-Lim の反復回数（増やすほど位相推定が良くなるが計算コスト増）
    """
    # mel -> linear-frequency STFT magnitude (pseudo-inverse)
    # librosa の mel_to_stft はデフォルトでパワー=2.0 の想定
    S = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft, power=power, n_mels=n_mels)

    # Griffin-Lim: S は振幅（またはパワーに応じたもの） -> 音声信号へ
    # librosa.griffinlim は「振幅スペクトル（non-power = magnitude）」を想定
    # mel_to_stft の出力 S がパワーなら np.sqrt して magnitude にする
    if power == 2.0:
        magnitude = np.sqrt(np.maximum(S, 1e-10))
    else:
        magnitude = S  # power==1 -> already magnitude

    y = librosa.griffinlim(magnitude, n_iter=n_iter, hop_length=hop_length, win_length=win_length, window='hann')
    return y

# 使い方の例（実際は mel_power をモデル出力で置き換えてください）
# y = mel_to_wave_via_griffinlim(mel_power)
# sf.write('reconstructed.wav', y, sr)