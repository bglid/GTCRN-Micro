# Copyright (c) 2024 urgent2024_challenge.
# Licensed under the Apache License, Version 2.0.

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from p_tqdm import p_map
from pesq import PesqError, pesq
from pystoi import stoi

METRICS = ("SDR", "SISNR", "PESQ", "STOI")


################################################################
# Definition of metrics
################################################################
def stoi_metric(ref: NDArray[Any], inf: NDArray[Any], fs: int = 16000):
    """Calculate Short-Time Objective Intelligibility (ESTOI).

    Args:
        ref (NDArray[Any]): reference signal (time,)
        inf (NDArray[Any]): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        stoi (float): STOI value between [0, 1]
    """
    return stoi(ref, inf, fs_sig=fs, extended=False)


def pesq_metric(ref: NDArray[Any], inf: NDArray[Any], fs: int = 8000):
    """Calculate Perceptual Evaluation of Speech Quality (PESQ).

    Args:
        ref (NDArray[Any]): reference signal (time,)
        inf (NDArray[Any]): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        pesq (float): PESQ value between [-0.5, 4.5]
    Raises:
        ValueError: Sample rate error if not 8kHz of 16kHz
    """
    assert ref.shape == inf.shape
    if fs == 8000:
        mode = "nb"
    elif fs == 16000:
        mode = "wb"
    elif fs > 16000:
        mode = "wb"
        ref = librosa.resample(ref, orig_sr=fs, target_sr=16000)
        inf = librosa.resample(inf, orig_sr=fs, target_sr=16000)
        fs = 16000
    else:
        raise ValueError(
            f"sample rate must be 8000 or 16000+ for PESQ evaluation, but got {fs}"
        )
    pesq_score = pesq(
        fs,
        ref,
        inf,
        mode=mode,
        on_error=PesqError.RETURN_VALUES,
    )
    if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
        logging.warning("[PESQ] Error: No utterances detected. Skipping this sample.")
    else:
        return pesq_score


def sisnr_metric(ref, inf):
    inf = inf - inf.mean()
    ref = ref - ref.mean()

    a = np.sum(inf * ref) / np.sum(ref**2 + 1e-8)
    e_tagt = a * ref
    e_res = inf - e_tagt

    return 10 * np.log10((np.sum(e_tagt**2) + 1e-8) / (np.sum(e_res**2) + 1e-8))


def sdr_metric(ref, inf):
    inf = inf - inf.mean()
    ref = ref - ref.mean()
    e_tagt = ref
    e_res = inf - e_tagt

    return 10 * np.log10((np.sum(e_tagt**2) + 1e-8) / (np.sum(e_res**2) + 1e-8))


################################################################
# Main entry
################################################################
def main(args):
    refs = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, refs[uid], audio_path))

    ret = []

    ### Single thread
    # for data_pair in tqdm(data_pairs):
    #     tmp = process_one_pair(data_pair)
    #     ret.append(tmp)

    ### Multi thread
    ret = p_map(
        process_one_pair,
        data_pairs,
        num_cpus=args.nj,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {metric: (outdir / f"{metric}.scp").open("w") for metric in METRICS}

    for uid, score in ret:
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    with (outdir / "RESULTS.txt").open("w") as f:
        for metric in METRICS:
            mean_score = np.nanmean([score[metric] for uid, score in ret])
            f.write(f"{metric}: {mean_score:.4f}\n")
    print(f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True)


def process_one_pair(data_pair):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    scores = {}
    for metric in METRICS:
        if metric == "PESQ":
            pesq_score = pesq_metric(ref, inf, fs=fs)
            scores[metric] = pesq_score if pesq_score is not None else np.nan
        elif metric == "STOI":
            scores[metric] = stoi_metric(ref, inf, fs=fs)
        elif metric == "SISNR":
            scores[metric] = sisnr_metric(ref, inf)
        elif metric == "SDR":
            scores[metric] = sdr_metric(ref, inf)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        help="Path to the scp file containing reference signals",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=8,
        help="Number of parallel workers to speed up evaluation",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Chunk size used in process_map",
    )
    args = parser.parse_args()

    main(args)
