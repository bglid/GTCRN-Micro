import numpy as np
import onnxruntime
import soundfile as sf
import tensorflow as tf
import torch

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def output_test() -> None:
    # loading data
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
        dtype="float32",
    )
    assert fs == 16000, f"Expected fs of 16000, instead got {fs}"
    # running stft
    input = torch.stft(
        torch.from_numpy(mix),
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )

    input = input[None][0]
    input = input[:, :63, :]
    # Check PyTorch output
    # load state dict from checkpoint
    model = GTCRNMicro()
    ckpt = torch.load(
        "./gtcrn_micro/ckpts/best_model_dns3.tar",
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
    print(f"\tmissing keys: {missing}")
    print(f"\tunexpected keys: {unexpected}")

    # explicitly setting model to eval in function
    model.eval()
    model.to("cpu")

    input = torch.randn(1, 257, 63, 2)
    with torch.no_grad():
        pytorch_output = model(input).cpu().numpy()
    print("\npytorch output done")

    # check onnx output
    session = onnxruntime.InferenceSession("./gtcrn_micro/models/onnx/gtcrn_micro.onnx")
    onnx_output = session.run(
        ["mask"],
        {
            "audio": input.numpy(),
        },
    )
    print("\nonnx output done")

    ## Load tflite model and compare outputs
    tflite_path = "./gtcrn_micro/models/tflite/gtcrn_micro_full_integer_quant.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # fix input shapes

    # input_shape1 = input_details[0]["shape"]
    input_data1 = input.permute(0, 2, 3, 1).detach().numpy().astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], input_data1)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    print("\ntflite output done")

    print(
        f"Onnx outputs error vs pytorch: {np.mean(np.abs(onnx_output[0] - pytorch_output))}"
    )
    diff = onnx_output[0] - pytorch_output
    abs_diff = np.abs(diff)

    print("MAE:", abs_diff.mean())
    print("Median abs diff:", np.median(abs_diff))
    print("95th percentile:", np.percentile(abs_diff, 95))
    print("99th percentile:", np.percentile(abs_diff, 99))
    print("Fraction > 0.5:", np.mean(abs_diff > 0.5))
    print("Fraction > 1.0:", np.mean(abs_diff > 1.0))
    print(
        f"Tflite outputs error vs pytorch: {np.mean(np.abs(pytorch_output - output_data))}"
    )
    print(
        f"Tflite outputs error vs onnx: {np.mean(np.abs(onnx_output[0] - output_data))}"
    )

    print("DONE")


if __name__ == "__main__":
    output_test()
