import torch
from gtcrn_micro.models.gtcrn_micro import GTCRNMicro


def test_gtcrn_micro():
    # causal test for workflow
    model = GTCRNMicro().eval()

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    x1 = torch.stft(
        x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
    )
    x2 = torch.stft(
        x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
    )
    y1 = model(x1)[0]
    y2 = model(x2)[0]

    # apparently newer torch wants complex
    y1 = torch.view_as_complex(y1.contiguous())
    y2 = torch.view_as_complex(y2.contiguous())

    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5))
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5))

    # y1 = y1.squeeze(0)
    # y2 = y2.squeeze(0)

    res1 = (y1[: 16000 - 256 * 2] - y2[: 16000 - 256 * 2]).abs().max()
    res2 = (y1[16000:] - y2[16000:]).abs().max()
    assert res1.item() == 0
    assert res2.item() > 0
