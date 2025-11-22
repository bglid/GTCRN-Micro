import torch
import numpy as np
import tensorflow as tf
import soundfile as sf


# function for creating dataset
def representative_data_gen(input):
    # need to fix axis dims
    audio = np.transpose(input, (1, 2, 0)).astype(np.float32)

    # need to add batch dimension
    audio = np.expand_dims(audio, axis=0)
    print(f"rep data shape: {audio.shape}")

    def rep_data():
        # for input_value in tf.data.Dataset.from_tensor_slices(input).take(100):

        # fixing batch dim
        for _ in range(100):
            yield [audio]

    return rep_data


# function for creating quantized model
def model_quantizer(model_path, representative_data_gen):
    # TFLM conversion to int8
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # DEBUG
    converter.representative_dataset = representative_data_gen
    # # check for ops errors in quantizations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    # # set input and output to int8
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    print("\n------\nStarting int8 conversion:\n")
    tflite_quant_model = converter.convert()
    print(
        f"[model_quantizer] converter.convert() finished, size in bytes: {len(tflite_quant_model)}"
    )

    return tflite_quant_model


if __name__ == "__main__":
    # loading model
    model_path = "./gtcrn_micro/models/tflite/"

    # # debug
    # converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    # tflite_32 = converter.convert()
    # print("Float32 TFLite conversion OK, size:", len(tflite_32))

    # DEBUG
    loaded = tf.saved_model.load(model_path)
    print("Signatures:", list(loaded.signatures.keys()))

    infer = loaded.signatures["serving_default"]  # usually this key
    print("Input signature:", infer.structured_input_signature)

    # creating sample rep data
    mix, fs = sf.read(
        "./gtcrn_micro/data/DNS3/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
        dtype="float32",
    )

    # running stft
    input = torch.stft(
        torch.from_numpy(mix),
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )

    print(f"STFT shape: {input.shape}")
    # get rep data dimension resolution
    rep_data = representative_data_gen(input.numpy())

    tflite_quant_model = model_quantizer(model_path, rep_data)

    with open(
        "./gtcrn_micro/models/tflite/gtcrn_micro_int8.tflite",
        "wb",
    ) as f:
        f.write(tflite_quant_model)
        print("int8 written!")
