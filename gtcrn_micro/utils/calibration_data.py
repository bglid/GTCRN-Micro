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
    # stft_np = stft.numpy().transpose(1, 2, 0)
    # fixing for TF NHWC
    stft_np = stft.numpy().transpose(1, 0, 2)  # -> (T, F, 2)
    return stft_np


def main():
    """Generate calibration data set by input wav."""
    # getting .wav files in directory
    wavs = sorted(CALIB_DATA.glob("*.wav"))[:500]
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
        T, F, C = tsr.shape

        assert C == 2, tsr.shape
        if T >= max_frames:
            tsr = tsr[:max_frames, :, :]
        # else do padding
        else:
            pad = np.zeros((max_frames - T, F, C), dtype=tsr.dtype)
            tsr = np.concatenate([tsr, pad], axis=0)

        padding.append(tsr)

    # reconstructing the now padded data (if needed)
    data = np.stack(padding, axis=0).astype(np.float32)

    print("**********\nData info\n**********")
    # debugging:
    print(data.shape, data.dtype)
    print(f"Min/max: {data.min(), data.max()}")
    print(f"po1/p99: {np.percentile(data, 1), np.percentile(data, 99)}")
    print("**********")

    # clipping data for quantization
    scale_low = np.percentile(data, 0.1)
    scale_high = np.percentile(data, 99.9)
    scale = max(abs(scale_low), abs(scale_high)) * 2.0

    clipped_data = np.clip(data / scale + 0.5, 0.0, 1.0).astype(np.float32)

    np.save(OUTPUT, clipped_data)
    print(f"Scale = {scale}")
    print("**********\nCalibration info\n**********")
    x = np.load(OUTPUT)
    print(f"Shape = {x.shape}")
    print(f"Min/max: {x.min(), x.max()}")
    print(f"po1/p99: {np.percentile(x, 1), np.percentile(x, 99)}")
    print("**********")


if __name__ == "__main__":
    main()
