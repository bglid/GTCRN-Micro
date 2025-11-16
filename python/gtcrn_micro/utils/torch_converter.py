import torch
import ai_edge_torch
import soundfile as sf

# from models.gtcrn.gtcrn import GTCRN
from models.GTCRN_micro import GTCRN_micro

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


model = GTCRN_micro().eval()
# missing, unexpected = model.load_state_dict(state, strict=False)
# print("Missing:", missing[:10], " ... total:", len(missing))
# print("Unexpected:", unexpected[:10], " ... total:", len(unexpected), "\n")
# model.eval()

# testing that forward pass works!
# loading test
mix, fs = sf.read(
    "./data/DNS3/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k/nonenglish_synthetic_male_SNR_21.0dB_german_1.wav",
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

# recovering audio test
# enhanced_audio = torch.istft(
#     y,
#     512,
#     256,
#     512,
#     torch.hann_window(512).pow(0.5),
#     return_complex=False,
# )
# sf.write("enhanced.wav", enhanced_audio.detach().cpu().numpy(), fs)

# conversion test
edge_model = ai_edge_torch.convert(model, (input[None],))
print("Conversion works!")

# test TFLite Micro model
edge_out = edge_model(
    input[None],
)
if hasattr(edge_out, "shape"):
    print(f"TFLM forward good! \n{edge_out.shape}")
else:
    print(type(edge_out))

# export flatbuffer
edge_model.export("gtcrn.tflite")
print("Export of flatbuffer worked!")
