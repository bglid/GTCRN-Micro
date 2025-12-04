# Source: Adpated from - https://github.com/Xiaobin-Rong/SEtrain
import os

import librosa
import numpy as np
import soundfile as sf

# import shutil
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro as Model


# NEEDED FOR DNS3
def _extract_fileid_from_noisy(path: str):
    """Grab the fileid substring from the *noisy* filename."""
    base = os.path.basename(path)
    if "fileid_" not in base:
        return None
    return base.split("fileid_")[-1].split(".")[0]


# for bulk inference
def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_network = OmegaConf.load(cfg_infer.network.config)

    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = Model(**cfg_network["network_config"]).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    noisy_wavs = sorted(
        list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder)))
    )

    inf_scp_list = []
    ref_scp_list = []
    for wav_name in tqdm(noisy_wavs):
        # getting path for fixing file matching
        noisy_path = os.path.join(noisy_folder, wav_name)
        # converting audio to sample rate if needed
        noisy, fs = sf.read(os.path.join(noisy_folder, wav_name), dtype="float32")

        if fs != 16000:
            noisy = librosa.resample(y=noisy, orig_sr=fs, target_sr=16000)
            fs = 16000
        assert fs == 16000

        # inference
        input = torch.stft(
            torch.from_numpy(noisy),
            512,
            256,
            512,
            torch.hann_window(512).pow(0.5),
            return_complex=False,
        )

        # input = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(input[None])[0]
        # enhanced = output.cpu().detach().numpy().squeeze()
        output = torch.view_as_complex(output.contiguous())
        enhanced = torch.istft(
            output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
        )

        # fixing paths
        fileid = _extract_fileid_from_noisy(noisy_path)
        if fileid is None:
            raise RuntimeError(f"Unable to extract: {noisy_path}")

        clean_name = f"clean_fileid_{fileid}.wav"
        ref_path = os.path.join(clean_folder, clean_name)

        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"Clean file not found for {clean_name}, fileid={fileid}:\n {ref_path}"
            )

        clean, fs_clean = sf.read(ref_path, dtype="float32")
        if fs_clean != fs:
            clean = librosa.resample(y=clean, orig_sr=fs_clean, target_sr=fs)
            fs_clean = fs
        assert fs_clean == fs

        # fixing shape issues
        if enhanced.shape[0] < clean.shape[0]:
            pad = clean.shape[0] - enhanced.shape[0]
            enhanced = np.pad(enhanced, (0, pad), mode="constant")
        elif enhanced.shape[0] > clean.shape[0]:
            enhanced = enhanced[: clean.shape[0]]

        uid = wav_name.split(".wav")[0]
        enh_path = os.path.join(enh_folder, uid + "_enh.wav")

        sf.write(enh_path, enhanced, fs)

        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, ref_path])

    # Save paths into scp file for evaluation
    with open(os.path.join(enh_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(enh_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # running from root
    parser.add_argument("-C", "--config", default="gtcrn_micro/conf/cfg_infer.yaml")
    parser.add_argument("-D", "--device", default="0", help="Index of the gpu device")

    args = parser.parse_args()
    main(args)
