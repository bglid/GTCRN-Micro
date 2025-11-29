# Original author Xiaobin Rong
# Source: SEtrain: https://github.com/Xiaobin-Rong/SEtrain
# Shoutout SEtrain
import os
from random import sample

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils import data
from tqdm import tqdm

# loading up VCTK dataset
# train_data_path = "/data/VCTK-DEMAND/noisy_testset_wav/"
# valid_data_path = ... # 1572 utt

# need to switch on HPC
DNS3_TRAIN = "./gtcrn_micro/data/DNS3/noisy"
DNS3_VALID = "./gtcrn_micro/data/DNS3_val/noisy"


# ------
class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs: int = 16000,
        length_seconds: int = 8,
        total_train_data: int = 180000,  # fix to shuffle better
        num_data_per_epoch: int = 40000,
        random_start: bool = False,
        train: bool = True,
    ):
        if train:
            print(f"Running Training on DNS3:\n{DNS3_TRAIN}")
        else:
            print(f"Running Validation on DNS3:\n{DNS3_VALID}")

        # Member variables for dataset
        self.noisy_database_train = sorted(
            librosa.util.find_files(DNS3_TRAIN, ext="wav")
        )[:total_train_data]
        # remaining data
        self.noisy_database_valid = sorted(
            librosa.util.find_files(DNS3_VALID, ext="wav")
        )
        self.length_samples = int(length_seconds * fs)
        self.random_start = random_start
        self.fs = fs
        self.length_seconds = length_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train

    def sample_data_per_epoch(self):
        self.noisy_data_train = sample(
            self.noisy_database_train, self.num_data_per_epoch
        )

    def __getitem__(self, index):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        noisy_path = noisy_list[index]
        noisy_name = os.path.basename(noisy_path)
        fileid = noisy_name.split("fileid_")[-1].split(".")[0]

        # build clean path
        clean_root = DNS3_TRAIN.replace("noisy", "clean")
        clean_name = f"clean_fileid_{fileid}.wav"
        clean_path = os.path.join(clean_root, clean_name)
        # clean_path = noisy_path.replace("noisy", "clean")

        try:
            if self.random_start:
                begin_s = int(np.random.uniform(0, 10 - self.length_seconds)) * self.fs
                noisy, _ = sf.read(
                    noisy_path,
                    dtype="float32",
                    start=begin_s,
                    stop=begin_s + self.length_samples,
                )
                clean, _ = sf.read(
                    clean_path,
                    dtype="float32",
                    start=begin_s,
                    stop=begin_s + self.length_samples,
                )
            else:
                noisy, _ = sf.read(
                    noisy_path,
                    dtype="float32",
                    start=0,
                    stop=self.length_samples,
                )
                clean, _ = sf.read(
                    clean_path,
                    dtype="float32",
                    start=0,
                    stop=self.length_samples,
                )
        except sf.LibsndfileError as e:
            print(
                f"LibsndfileError on:\n  noisy={noisy_path}\n  clean={clean_path}\n  {e}",
                flush=True,
            )
            raise

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)


if __name__ == "__main__":
    # conf = OmegaConf.load("gtcrn_micro/conf/cfg_train_VCTK.yaml")
    conf = OmegaConf.load("gtcrn_micro/conf/cfg_train_DNS3.yaml")

    train_dataset = DNS3Dataset(**conf["train_dataset"])
    train_dataloader = data.DataLoader(train_dataset, **conf["train_dataloader"])
    train_dataloader.dataset.sample_data_per_epoch()

    valid_dataset = DNS3Dataset(**conf["valid_dataset"])
    valid_dataloader = data.DataLoader(valid_dataset, **conf["valid_dataloader"])

    print(f"Length of train dataloader: {len(train_dataloader)}\n")
    print(f"Length of valid dataloader: {len(valid_dataloader)}\n")

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break

    for noisy, clean in tqdm(valid_dataloader):
        print(noisy.shape, clean.shape)
        break
