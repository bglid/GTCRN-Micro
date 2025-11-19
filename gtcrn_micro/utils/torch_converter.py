import torch

# import ai_edge_torch
import soundfile as sf
import onnx

# test
from torch import export

# from models.gtcrn.gtcrn import GTCRN
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


model = GTCRNMicro().eval()


# testing that forward pass works!
# loading test
mix, fs = sf.read(
    "./gtcrn_micro/data/DNS3/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
    dtype="float32",
)
print("\n", mix)
print(fs)
assert fs == 16000
# running stft
input = torch.stft(
    torch.from_numpy(mix),
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

# # recovering audio test
# y = torch.view_as_complex(y.contiguous())
# enhanced_audio = torch.istft(
#     y,
#     512,
#     256,
#     512,
#     torch.hann_window(512).pow(0.5),
# )
# sf.write("enhanced.wav", enhanced_audio.detach().cpu().numpy(), fs)

# making a smaller input for conversion
input_small = input[:, :64, :]

# test export from torch
exported = export.export(model, (input_small[None],))
print("torch export works...")


# -----------------------
# Testing out Torch -> ONNX -> TF -> TFLM
# onnx_program = torch.onnx.export(model, (input[None]), dynamo=True, report=True)
# onnx_program.save("gtcrn_micro.onnx")

# older approach - Works!
torch.onnx.export(
    model,
    (input[None]),
    "gtcrn_micro.onnx",
    opset_version=16,  # Lowerin opset for LN
    dynamo=False,
    input_names=["audio"],
    output_names=["mask"],
    export_params=True,
    do_constant_folding=True,
    report=True,
)


# checking the model
onnx_model = onnx.load("gtcrn_micro.onnx")
onnx.checker.check_model(onnx_model)
