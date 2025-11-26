# Shoutout SEtrain
from random import sample

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils import data
from tqdm import tqdm

# loading up VCTK dataset
train_data_path = "/data/VCTK-DEMAND/noisy_testset_wav/"
# valid_data_path = ... # 1572 utt

# need to switch on HPC
DNS3_TRAIN = "./data/DNS3/V2_V3_DNSChallenge_Blindset/"
DNS3_VALID = "./data/DNS3/V2_V3_DNSChallenge_Blindset/"


# Class to define VCTK dataset
class VCTKDataset(torch.utils.data.Dataset):
    # may need to adjust args after unpacking data in HPC
    def __init__(
        self,
        fs: int = 16000,
        # length_seconds: int=8,
        total_train_data: int = 10000,  # fix to shuffle better
        # num_data_per_epoch: int=
        random_start: bool = False,
        train: bool = True,
    ):
        if train:
            print(f"Running Training on VCTK-DEMAND:\n{train_data_path}")
        else:
            # print(f"Running Validation on VCTK-DEMAND:\n{valid_data_path}")
            print("Running Validation on VCTK-DEMAND")

        # Member variables for dataset
        self.noisy_data_train = sorted(
            librosa.util.find_files(train_data_path, ext="wav")
        )[:total_train_data]
        # remaining data
        self.noisy_data_valid = sorted(
            librosa.util.find_files(train_data_path, ext="wav")
        )[total_train_data:]
        # self.length_samples = int(length_seconds * fs)
        self.fs = fs
        self.random_start = random_start
        # self.length_seconds = length_seconds
        # self.num_data_per_epoch = num_data_per_epoch
        self.train = train

    def __getitem__(self, index):
        if self.train:
            noise_list = self.noisy_data_train
        else:
            noise_list = self.noisy_data_valid

        # data is all different lengths in VCTK Dataset
        if self.random_start:
            begin_s = int(np.random.uniform(0, 1)) * self.fs
            noisy, sr = sf.read(noise_list[index], dtype="float32", start=begin_s)
            clean, sr = sf.read(
                noise_list[index].replace("noisy", "clean"),
                dtype="float32",
                start=begin_s,
            )
        else:
            noisy, sr = sf.read(noise_list[index], dtype="float32")
            clean, sr = sf.read(
                noise_list[index].replace("noisy", "clean"),
                dtype="float32",
            )

        return noisy, clean


# ------
class DNS3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fs: int = 16000,
        length_seconds: int = 8,
        total_train_data: int = 720000,  # fix to shuffle better
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

        # setting random starting point
        if self.random_start:
            begin_s = int(np.random.uniform(0, 10 - self.length_seconds)) * self.fs
            noisy, _ = sf.read(
                noisy_list[idx],
                dtype="float32",
                start=begin_s,
                stop=begin_s + self.length_samples,
            )
            clean, _ = sf.read(
                noisy_list[idx].replace("noisy", "clean"),
                dtype="float32",
                start=begin_s,
                stop=begin_s + self.length_samples,
            )

        else:
            noisy, _ = sf.read(
                noisy_list[idx],
                dtype="float32",
                start=0,
                stop=self.length_samples,
            )
            clean, _ = sf.read(
                noisy_list[idx].replace("noisy", "clean"),
                dtype="float32",
                start=0,
                stop=self.length_samples,
            )

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
