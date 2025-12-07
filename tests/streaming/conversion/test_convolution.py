import torch
import torch.nn as nn
from numpy.testing import assert_allclose

from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.conversion.convolution import (
    StreamConv1d,
    StreamConv2d,
    StreamConvTranspose2d,
)


def test_convolution():
    # running test cases
    # Conv1d Stream
    Sconv = StreamConv1d(1, 1, 3)
    Conv = nn.Conv1d(1, 1, 3)
    convert_to_stream(stream_model=Sconv, model=Conv)

    test_input = torch.randn([1, 1, 10])
    with torch.no_grad():
        # non-streaming
        test_out1 = Conv(torch.nn.functional.pad(test_input, [2, 0]))

        # streaming
        cache = torch.zeros([1, 1, 2])
        test_out2 = []
        for i in range(10):
            # passing each k-1 cache output to the next
            out, cache = Sconv(test_input[..., i : i + 1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=-1)
        assert_allclose(((test_out1 - test_out2).abs().max()), 0, atol=1e-6)

    # Conv2d Stream
    Sconv = StreamConv2d(1, 1, 3)
    Conv = nn.Conv2d(1, 1, 3)
    convert_to_stream(stream_model=Sconv, model=Conv)

    test_input = torch.randn([1, 1, 10, 6])
    with torch.no_grad():
        # non-streaming
        test_out1 = Conv(torch.nn.functional.pad(test_input, [0, 0, 2, 0]))

        # streaming
        cache = torch.zeros([1, 1, 2, 6])
        test_out2 = []
        for i in range(10):
            # passing each k-1 cache output to the next
            out, cache = Sconv(test_input[:, :, i : i + 1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)
        assert_allclose(((test_out1 - test_out2).abs().max()), 0, atol=1e-6)

    # ConvTranspose2d Stream
    kt = 3  # kernel size along time axis
    dt = 2  # dilation along time axis
    pt = (kt - 1) * dt  # padding along time axis
    DeConv = torch.nn.ConvTranspose2d(
        4, 8, (kt, 1), stride=(1, 1), padding=(pt, 1), dilation=(dt, 2), groups=1
    )
    # NOTE: This fails based on the assert self.T_pad == 0
    # SDeConv = StreamConvTranspose2d(
    #     4, 8, (kt, 3), stride=(1, 2), padding=(2 * 2, 1), dilation=(dt, 2), groups=2
    # )

    # updated streaming padding
    SDeConv = StreamConvTranspose2d(
        4, 8, (kt, 1), stride=(1, 1), padding=(0, 1), dilation=(dt, 2), groups=1
    )
    convert_to_stream(SDeConv, DeConv)

    test_input = torch.randn([1, 4, 100, 6])
    with torch.no_grad():
        # non streaming
        test_out1 = DeConv(nn.functional.pad(test_input, [0, 0, pt, 0]))

        # streaming
        test_out2 = []
        cache = torch.zeros([1, 4, pt, 6])
        for i in range(100):
            out, cache = SDeConv(test_input[:, :, i : i + 1], cache)
            test_out2.append(out)
        test_out2 = torch.cat(test_out2, dim=2)
        assert_allclose(((test_out1 - test_out2).abs().max()), 0, atol=1e-6)
