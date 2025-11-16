import torch
from torch import float64, randn
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalNoiseRatio,
)


def full_evaluation(preds: torch.Tensor, target: torch.Tensor) -> dict[str, float64]:
    """Runs full suite of intrusive eval metrics for SE (STOI, PESQ, SISNR)

    Args:
        preds (torch.Tensor): Predictions, enhanced speech from model
        target (torch.Tensor): Target clean speech

    Returns:
        dict: Dictionary of mean result for each metric
    """
    # Init dict for results
    metric_res = {
        "STOI": 0.0,
        "PESQ": 0.0,
        "SISNR": 0.0,
    }

    # stoi test
    stoi = ShortTimeObjectiveIntelligibility(8000, False)
    metric_res["STOI"] = stoi(preds, target)
    print(f"STOI Test Results: \n {metric_res['STOI']}")

    # pesq test
    pesq = PerceptualEvaluationSpeechQuality(8000, "nb")
    metric_res["PESQ"] = pesq(preds, target)
    print(f"PESQ Test Results: \n {metric_res['PESQ']}")

    # sisnr test
    sisnr = ScaleInvariantSignalNoiseRatio()
    metric_res["SISNR"] = sisnr(preds, target)
    print(f"SISNR Test Results: \n {metric_res['SISNR']}")

    # getting the results for each type
    return metric_res


if __name__ == "__main__":
    # test
    preds = randn(8000)
    target = randn(8000)
