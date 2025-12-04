import numpy as np
import onnx
import soundfile as sf
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import export

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def torch2onnx(
    model: nn.Module,
    sample_input: NDArray[np.float64],
    time_chunk: int,
    model_name: str,
    checkpoint: str,
) -> None:
    """Convert Torch model to .onnx.

    Args:
        model (nn.Module): Model to convert to onnx
        sample_input (NDArray[np.float64]): Sample small input for conversion
        time_chunk (int): Time in samples for the amount of audio you want for your input
        model_name (str): Name of onnx file that will be saved - "name".onnx
        checkpoint (str): Path to the model checkpoint for conversion
    """
    ONNX_PATH = "./gtcrn_micro/models/onnx/"

    # loading up model checkpoints
    ckpt = torch.load(
        checkpoint,
        map_location="cpu",
    )

    state = (
        ckpt.get("state_dict", None)
        or ckpt.get("model_state_dict", None)
        or ckpt.get("model", None)
        or ckpt
    )

    # Handling if ckpt was saved from DDP and has module prefixes
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    # print state dict info
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("-" * 20)
    print(f"\nLoaded checkpoint: {checkpoint}")
    print(f"\tmissing keys: {missing}")
    print(f"\tunexpected keys: {unexpected}")

    # explicitly setting model to eval in function
    model.eval()
    model.to("cpu")

    # testing that forward pass works!
    assert fs == 16000, f"Expected fs of 16000, instead got {fs}"
    # running stft
    input = torch.stft(
        torch.from_numpy(sample_input),
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )

    # forward pass test
    with torch.no_grad():
        y = model(input[None])[0]
    print("Forward works!", tuple(y.shape) if hasattr(y, "shape") else type(y), "\n")

    # making a smaller input for conversion
    input_small = input[:, :time_chunk, :]
    print(f"input small size: {input_small.shape}")

    # test export from torch
    export.export(model, (input_small[None],))
    print("torch export works...")

    # -----------------------
    # Torch -> ONNX for later -> TF -> TFLM

    print("starting onnx export:")
    torch.onnx.export(
        model,
        (input_small[None]),  # Exporting with small input
        f"{ONNX_PATH}{model_name}.onnx",
        opset_version=16,  # Lowerin opset for LN
        dynamo=False,
        input_names=["audio"],
        output_names=["mask"],
        dynamic_axes=None,
        export_params=True,
        do_constant_folding=True,
        report=True,
    )

    # checking the model
    onnx_model = onnx.load(f"{ONNX_PATH}{model_name}.onnx")
    onnx.checker.check_model(onnx_model)
    # print ONNX input shape
    print("Onnx input shape:\n----------")
    for i in onnx_model.graph.input:
        print("\n", i.type.tensor_type.shape)


if __name__ == "__main__":
    # loading model
    model = GTCRNMicro()

    # loading test data
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
        dtype="float32",
    )

    ckpt = "./gtcrn_micro/ckpts/best_model_dns3.tar"
    # converting model, 1 second time chunk
    torch2onnx(model, mix, time_chunk=63, model_name="gtcrn_micro", checkpoint=ckpt)
