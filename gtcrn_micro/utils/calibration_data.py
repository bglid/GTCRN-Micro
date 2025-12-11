# used for onnx2tf.sh
# ----------------
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

# starting with basic data calibration
# NOTE: Need to improve in the future
CALIB_DATA = Path(
    "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/"
)
OUTPUT = Path("./gtcrn_micro/models/tflite/tflite_calibration.npy")

# CONSTANTS FOR STFT INFO
N_FFT = 512
HOP = 256


# function to generate stft tensors
def wav_2_tensor(wav_path: Path) -> NDArray[np.float64]:
    """Transform input to np tensor for model input.

    Args:
        wav_path (Path): Path to input wavs

    Returns:
        NDArray[np.float64]: STFT transformed input wav

    """
    mix, fs = sf.read(
        str(wav_path),
        dtype="float32",
    )

    assert fs == 16000, f"Expected 16kHz, got {fs}"

    stft = torch.stft(
        torch.from_numpy(mix),
        N_FFT,
        HOP,
        N_FFT,
        torch.hann_window(N_FFT).pow(0.5),
        return_complex=False,
    )

    # specifically for the tf model conversion we need to transpose the inputs
    stft_np = stft.numpy().transpose(1, 2, 0)
    return stft_np


def main():
    """Generate calibration data set by input wav."""
    # getting .wav files in directory
    # only taking 32 samples for now
    wavs = sorted(CALIB_DATA.glob("*.wav"))[:32]
    data = []
    # appending the tensor stft to data list
    for i in wavs:
        stft_np = wav_2_tensor(i)
        data.append(stft_np)

    # need to pad or truncate data
    # max frames comes from previously working with conversion
    max_frames = 973
    # max_frames = 63  # changing max frames for lighter model
    padding = []
    # getting every tensor and checking it's shapes
    for tsr in data:
        T, C, F = tsr.shape
        if T >= max_frames:
            tsr = tsr[:max_frames]
        # else do padding
        else:
            pad = np.zeros((max_frames - T, C, F), dtype=tsr.dtype)
            tsr = np.concatenate([tsr, pad], axis=0)

        padding.append(tsr)

    # reconstructing the now padded data (if needed)
    data = np.stack(padding, axis=0)

    print(f"**********\nData shape: {data.shape}\n**********")
    np.save(OUTPUT, data.astype("float32"))


if __name__ == "__main__":
    main()
