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

    input = input[None, ...]
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
    # NOTE: not loading state-dict until new model is trained
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("-" * 20)
    print(f"\tmissing keys: {missing}")
    print(f"\tunexpected keys: {unexpected}")

    # explicitly setting model to eval in function
    model.eval()
    model.to("cpu")

    with torch.no_grad():
        pytorch_output = model(input).cpu().numpy()
    print("\npytorch output done")

    # check onnx output
    session = onnxruntime.InferenceSession(
        "./gtcrn_micro/streaming/onnx/gtcrn_micro.onnx"
    )
    onnx_output = session.run(
        ["mask"],
        {
            "audio": input.numpy(),
        },
    )
    print("\nonnx output done")

    # TFLite
    ## Load tflite model and compare outputs
    tflite_path = "./gtcrn_micro/streaming/tflite/gtcrn_micro_full_integer_quant.tflite"
    # tflite_path = "./gtcrn_micro/streaming/tflite/gtcrn_micro_float32.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    input_data1 = input.permute(0, 2, 3, 1).detach().numpy().astype(np.float32)

    print("\n------------\nTFLite shape check:\n")
    print(">>INPUTS<<")
    for d in interpreter.get_input_details():
        print(f"{d['name']}, shape: {d['shape']}\ndtype: {d['dtype']}")

    for d in interpreter.get_output_details():
        print("\n>>OUTPUTS<<")
        print(f"{d['name']}, shape: {d['shape']}\ndtype: {d['dtype']}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(
        input_details[0]["index"], input_data1.shape, strict=True
    )
    interpreter.allocate_tensors()
    print(
        "in quant:",
        input_details[0]["quantization"],
        input_details[0]["quantization_parameters"],
    )
    print(
        "out quant:",
        output_details[0]["quantization"],
        output_details[0]["quantization_parameters"],
    )

    # fix input scale
    in_scale, in_zero = input_details[0]["quantization"]
    out_scale, out_zero = output_details[0]["quantization"]

    # interpreter.resize_tensor_input(
    #     input_details[0]["index"], input_data1.shape, strict=True
    # )
    #
    # interpreter.allocate_tensors()
    # setting input data to match the input details shape and size
    if input_details[0]["dtype"] == np.int8:
        q = np.round(input_data1 / in_scale + in_zero)

        sat_lo = np.mean(q < -128)
        sat_hi = np.mean(q > 127)
        print(f"saturation lo%: {sat_lo * 100:.4f}  hi%: {sat_hi * 100:.4f}")

        float_min = input_data1.min()
        float_max = input_data1.max()
        q_float_min = (-128 - in_zero) * in_scale
        q_float_max = (127 - in_zero) * in_scale
        print("float min/max:", float_min, float_max)
        print("quant float range:", q_float_min, q_float_max)

        print("pre-clip q min/max:", q.min(), q.max())
        q = np.clip(q, -128, 127)
        x_q = q.astype(np.int8)
        # x_q = np.round(input_data1 / in_scale + in_zero).astype(np.int8)
    else:
        x_q = input_data1

    interpreter.set_tensor(input_details[0]["index"], x_q)
    interpreter.invoke()

    y_q = interpreter.get_tensor(output_details[0]["index"])

    print("\noutput shape: ", y_q.shape, "dtype: ", y_q.dtype)
    # dequantizing for comparison
    if output_details[0]["dtype"] == np.int8:
        tflite_output = (y_q.astype(np.float32) - out_zero) * out_scale
    else:
        tflite_output = y_q.astype(np.float32)

    print("\n-------------")

    print("\nTFLite output done\n")
    print("*" * 10)
    print("\nOUTPUT STATS\n")

    print(
        f"Onnx outputs error vs pytorch: {np.mean(np.abs(onnx_output[0] - pytorch_output))}"
    )
    diff_onnx = onnx_output[0] - pytorch_output
    abs_diff = np.abs(diff_onnx)

    print("onnx MAE:", abs_diff.mean())
    print("onnx median abs diff:", np.median(abs_diff))
    print(
        f"Tflite outputs error vs pytorch: {np.mean(np.abs(pytorch_output - tflite_output))}"
    )
    print(
        f"Tflite outputs error vs onnx: {np.mean(np.abs(onnx_output[0] - tflite_output))}"
    )
    diff_tflite = tflite_output - pytorch_output
    abs_diff_t = np.abs(diff_tflite)
    print("TFLite MAE:", abs_diff_t.mean())
    print("TFLite median abs diff:", np.median(abs_diff_t))

    print("DONE\n")
    print("*" * 10)


if __name__ == "__main__":
    output_test()
