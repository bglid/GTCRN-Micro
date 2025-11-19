import torch

# import numpy as np
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from ai_edge_litert.interpreter import Interpreter
import soundfile as sf


def model_comparison(input_wav) -> None:
    # running stft
    input = torch.stft(
        torch.from_numpy(mix),
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )

    # load GTCRN-Micro
    model = GTCRNMicro().eval()
    with torch.no_grad():
        torch_output = model(input[None])[0]
    print("[PyTorch] Model Predictions:", torch_output)

    # load onnx model
    # NOTE: Need to fix this:
    # session = onnxruntime.InferenceSession("gtcrn_micro/models/onnx/gtcrn_micro.onnx")
    # onnx_output = session.run(["mask"], {"audio": input[None]})
    # print("[ONNX] Model Predictions:", onnx_output)

    # TFLite model test
    interpreter = Interpreter(
        model_path="gtcrn_micro/models/tflite/gtcrn_micro_float32.tflite"
    )
    tf_lite_model = interpreter.get_signature_runner()
    inputs = {"audio": input[None][0]}

    tf_lite_output = tf_lite_model(**inputs)
    print("[TFLite] Model Predictions:", tf_lite_output)


if __name__ == "__main__":
    # test data setup
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
        dtype="float32",
    )
    # print("\n", mix)
    # print(fs)
    assert fs == 16000

    model_comparison(mix)
