from loss import Loss_for_test
from create_datasets import load_wav, crop_or_pad


def calculate_score(target_str, generated_str):

    target_wav = load_wav(target_str)
    target_wav = crop_or_pad(target_wav)
    generated_wav = load_wav(generated_str)
    generated_wav = crop_or_pad(generated_wav)

    mel_loss = Loss_for_test(
        target_wav, generated_wav, fft_size=1024, hop_size=256
    )
    return mel_loss


def test(targets, generated_str, test_group_name="A"):
    score = 0.0
    for target in targets:
        target_score = 0.0
        target_score = calculate_score(target, generated_str)
        score += target_score
    print(
        f"Final Score for Generated in {test_group_name} : {generated_str} is {score/len(targets)}"
    )


if __name__ == "__main__":
    A_group = [
        "datasets/C3/0001.wav",
        "datasets/C3/0002.wav",
        "datasets/C3/0003.wav",
        "datasets/C3/0004.wav",
        "datasets/C3/0005.wav",
        "datasets/C3/0006.wav",
        "datasets/C3/0007.wav",
        "datasets/C3/0008.wav",
        "datasets/C3/0009.wav",
        "datasets/C3/0010.wav",
        "datasets/C3/0011.wav",
    ]

    B_group = [
        "datasets/C3/0012.wav",
        "datasets/C3/0013.wav",
        "datasets/C3/0014.wav",
        "datasets/C3/0015.wav",
        "datasets/C3/0016.wav",
        "datasets/C3/0017.wav",
        "datasets/C3/0018.wav",
        "datasets/C3/0019.wav",
        "datasets/C3/0020.wav",
        "datasets/C3/0021.wav",
        "datasets/C3/0022.wav",
        "datasets/C3/0023.wav",
    ]

    C_group = [
        "datasets/C3/0024.wav",
        "datasets/C3/0025.wav",
        "datasets/C3/0026.wav",
        "datasets/C3/0027.wav",
        "datasets/C3/0028.wav",
        "datasets/C3/0029.wav",
        "datasets/C3/0030.wav",
        "datasets/C3/0031.wav",
        "datasets/C3/0032.wav",
        "datasets/C3/0033.wav",
        "datasets/C3/0034.wav",
        "datasets/C3/0035.wav",
        "datasets/C3/0036.wav",
    ]
    generated_str = "generated_ref.wav"
    test(A_group, generated_str, "A")
    test(B_group, generated_str, "B")
    test(C_group, generated_str, "C")
