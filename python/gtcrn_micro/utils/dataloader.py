#
import soundfile as sf
import librosa
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf


# loading up VCTK dataset
# do on HPC
# train_data_path = ...  # 10k utt
train_data_path = "/data/VCTK-DEMAND/noisy_testset_wav/"
# valid_data_path = ... # 1572 utt


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

    # randomizing train data
    # def sample_data_per_epoch(self):
    #     self.noisy_data_train = random.sample(self.noisy_data_train, self.num_data_per_epoch)

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


if __name__ == "__main__":
    conf = OmegaConf.load("conf/cfg_train.yaml")

    train_dataset = VCTKDataset(**conf["train_dataset"])
    train_dataloader = data.DataLoader(train_dataset, **conf["train_dataloader"])

    valid_dataset = VCTKDataset(**conf["valid_dataset"])
    valid_dataloader = data.DataLoader(valid_dataset, **conf["valid_dataloader"])

    print(f"Length of train dataloader: {len(train_dataloader)}\n")
    print(f"Length of valid dataloader: {len(valid_dataloader)}\n")

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break

    for noisy, clean in tqdm(valid_dataloader):
        print(noisy.shape, clean.shape)
        break
