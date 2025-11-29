import numpy as np
import onnx
import soundfile as sf
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import export

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro

# main entry
# def main() -> None:
#     parser = argparse.ArgumentParser()
#     # TODO, add args here
#     pass


# ckpt = torch.load(
#     # "./models/gtcrn/checkpoints/model_trained_on_vctk.tar", map_location="cpu"
#     "./models/gtcrn/checkpoints/model_trained_on_dns3.tar",
#     map_location="cpu",
# )

# state
# state = (
#     ckpt.get("state_dict", None)
#     or ckpt.get("model_state_dict", None)
#     or ckpt.get("model", None)
#     or ckpt
# )


def torch2onnx(
    model: nn.Module,
    sample_input: NDArray[np.float64],
    time_chunk: int,
    model_name: str,
) -> None:
    """Convert Torch model to .onnx.

    Args:
        model (nn.Module): Model to convert to onnx
        sample_input (NDArray[np.float64]): Sample small input for conversion
        time_chunk (int): Time in samples for the amount of audio you want for your input
        model_name (str): Name of onnx file that will be saved - "name".onnx
    """
    ONNX_PATH = "./gtcrn_micro/models/onnx/"
    # testing that forward pass works!
    assert fs == 16000
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
    # onnx_program = torch.onnx.export(model, (input[None]), dynamo=True, report=True)
    # onnx_program.save("gtcrn_micro.onnx")

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
    model = GTCRNMicro().eval()

    # loading test data
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
        dtype="float32",
    )

    # converting model
    torch2onnx(model, mix, time_chunk=63, model_name="gtcrn_micro")
