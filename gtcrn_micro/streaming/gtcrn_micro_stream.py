"""
GTCRN-Micro-Stream: MCU-focused rebuild of GTCRN, setup with streaming caching
"""

import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm

from gtcrn_micro.models.gtcrn_micro import GTCRNMicro
from gtcrn_micro.streaming.conversion.convert import convert_to_stream
from gtcrn_micro.streaming.conversion.convolution import (
    StreamConv2d,
    StreamConvTranspose2d,
)


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(
            erb_subband_1, erb_subband_2, nfft, high_lim, fs
        )
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(
        self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000
    ):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2] : bins[-1] + 1] = (
            1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
        )

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., : self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1 :])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


# Light SFE, depthwise 1x3 to gather local subband context
class SFE_Lite(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=in_channels,
            bias=False,
        )

    def forward(self, x):  # (B, 3, T, F)
        return self.depth_conv(x)


class TRALite(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int = 3,
    ) -> None:
        super().__init__()
        assert kernel >= 1
        self.channels = channels
        self.kernel = kernel
        self.L = kernel - 1

        self.depth_conv = nn.Conv1d(
            channels,
            channels,
            kernel,
            padding=0,
            groups=channels,
            bias=True,
        )
        self.point_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.act = nn.Sigmoid()

    def init_cache(self, B: int, device=None, dtype=None):
        if self.L == 0:
            return None
        return torch.zeros(B, self.channels, self.L, device=device, dtype=dtype)

    def forward(self, x, tra_cache):
        e = (x * x).mean(dim=-1)

        assert tra_cache is None or tra_cache.shape[1] == e.shape[1], (
            tra_cache.shape,
            e.shape,
        )
        assert tra_cache is None or tra_cache.shape[2] == self.L, (
            tra_cache.shape,
            self.L,
        )

        if self.L == 0:
            y = self.depth_conv(e)
            tra_cache = None
        else:
            if tra_cache is None:
                tra_cache = self.init_cache(x.size(0), device=x.device, dtype=x.dtype)

            e_cat = torch.cat([tra_cache, e], dim=2)
            y = self.depth_conv(e_cat)
            tra_cache = e_cat[:, :, -self.L :].contiguous()

        g = self.point_conv(y)
        g = self.act(g).unsqueeze(-1)

        return x * g, tra_cache


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        use_deconv=False,
        is_last=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # trying to fix conversion issue
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlock(nn.Module):
    """Streaming Group Temporal Convolution"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_deconv=False,
    ):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        stream_conv_module = StreamConvTranspose2d if use_deconv else StreamConv2d

        self.point_conv1 = conv_module(in_channels // 2, hidden_channels, 1)  # no SFE
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        if use_deconv:
            self.depth_conv = stream_conv_module(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                # groups=hidden_channels,
                groups=1,  # fixing for conversion
            )
        else:
            self.depth_conv = stream_conv_module(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                # groups=hidden_channels,
                groups=16,  # fixing for conversion
            )

        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)

        self.tra = TRALite(in_channels // 2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)."""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        # adjusting for streaming
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

        return x

    def forward(self, x, conv_cache, tra_cache):
        """
        x: (B, C, T, F)
        conv_cache: (B, C, (kT-1)*dT, F)
        """
        x1, x2 = x[:, : x.shape[1] // 2], x[:, x.shape[1] // 2 :]

        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        # streaming
        h1, conv_cache = self.depth_conv(h1, conv_cache)
        # bn + act on just h
        h1 = self.depth_act(self.depth_bn(h1))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1, tra_cache = self.tra(h1, tra_cache)
        x = self.shuffle(h1, x2)

        return x, conv_cache, tra_cache


class StreamTCN(nn.Module):
    """
    Streaming Temporal Convolutional Block.
    """

    def __init__(self, channels, kernel_size=3, dilation=1) -> None:
        super().__init__()
        # padding setup
        # self.pad = dilation * (kernel_size - 1)
        self.L = (kernel_size - 1) * dilation

        # conv1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU()

        # dep temporal conv
        self.conv2 = StreamConv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(0, 0),
            dilation=(dilation, 1),
            groups=channels,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.PReLU()

        # conv 3 brings us back to same size as conv1
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act3 = nn.PReLU()

    def forward(self, x, tcn_cache):
        """
        x: (batch, seq length, input size, freq bins)
        tcn_cache: (batch, seq length, input size: (k-1)*dilation, freq bins)
        """

        # we are going to use a residual connection within this block
        residual = x

        # first conv pass
        y1 = self.act1(self.bn1(self.conv1(x)))

        # Conv2 pass
        y2, tcn_cache = self.conv2(y1, tcn_cache)
        y2 = self.act2(self.bn2(y2))

        # conv 3 but doing the activation with the resid
        y3 = self.bn3(self.conv3(y2))

        res = y3 + residual
        return self.act3(res), tcn_cache


class StreamGTCN(nn.Module):
    """ """

    def __init__(self, channels, n_layers=4, kernel_size=3, dilation=2) -> None:
        super().__init__()
        # trying to stack into blocks to recreate dp
        blocks = []
        d = 1
        for i in range(n_layers):
            blocks.append(
                StreamTCN(channels=channels, kernel_size=kernel_size, dilation=d)
            )
            d *= dilation  # increases each block

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, tcn_cache):
        """
        x: (B, C, T, F).
        tcn_cache: (B, C, 2*d, F)
        """

        # new cache to hold for each layer
        new_cache = []
        for tcn, cache in zip(self.blocks, tcn_cache):
            x, cache = tcn(x, cache)
            new_cache.append(cache)

        return x, new_cache


class StreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(
                    3,  # no SFE
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    use_deconv=False,
                    is_last=False,
                ),
                ConvBlock(
                    16,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=1,
                    use_deconv=False,
                    is_last=False,
                ),
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=False,
                ),
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=False,
                ),
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=False,
                ),
            ]
        )

    def forward(self, x, conv_cache, tra_cache):
        en_outs = []
        # just passing as normal through first two conv blocks
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)

        # streaming forward pass for Streaming blocks
        # NOTE: due to dilation quant restrictions, need to make smaller dilation caches
        x, conv_cache[:, :, :2, :], tra_cache[0] = self.en_convs[2](
            x, conv_cache[:, :, :2, :], tra_cache[0]
        )
        en_outs.append(x)

        x, conv_cache[:, :, 2:4, :], tra_cache[1] = self.en_convs[3](
            x, conv_cache[:, :, 2:4, :], tra_cache[1]
        )
        en_outs.append(x)

        x, conv_cache[:, :, 4:6, :], tra_cache[2] = self.en_convs[4](
            x, conv_cache[:, :, 4:6, :], tra_cache[2]
        )
        en_outs.append(x)

        return x, en_outs, conv_cache, tra_cache


class StreamDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),  # causual padding!
                    dilation=(1, 1),
                    use_deconv=True,
                ),
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=True,
                ),
                StreamGTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=True,
                ),
                ConvBlock(
                    16,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=1,  # for TFLM conversion
                    use_deconv=True,
                    is_last=False,
                ),
                ConvBlock(
                    16,
                    2,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    use_deconv=True,
                    is_last=True,
                ),
            ]
        )

    def forward(self, x, en_outs, conv_cache, tra_cache):
        # decoding the cache backwards
        x, conv_cache[:, :, 4:6, :], tra_cache[0] = self.de_convs[0](
            x + en_outs[4], conv_cache[:, :, 4:6, :], tra_cache[0]
        )

        x, conv_cache[:, :, 2:4, :], tra_cache[1] = self.de_convs[1](
            x + en_outs[3], conv_cache[:, :, 2:4, :], tra_cache[1]
        )

        x, conv_cache[:, :, :2, :], tra_cache[2] = self.de_convs[2](
            x + en_outs[2], conv_cache[:, :, :2, :], tra_cache[2]
        )

        # iter as normal through last two conv blocks
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4 - i])
        return x, conv_cache, tra_cache


class Mask(nn.Module):
    """Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class StreamGTCRNMicro(nn.Module):
    def __init__(
        self,
        n_fft=512,  # just for config.yaml
        hop_len=256,
        win_len=512,
    ):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE_Lite(in_channels=3)

        self.encoder = StreamEncoder()

        self.gtcn1 = StreamGTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)
        self.gtcn2 = StreamGTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)

        self.decoder = StreamDecoder()

        self.mask = Mask()

    def forward(self, spec, conv_cache, tra_cache, tcn_cache):
        """
        spec: (B, F, T, 2)
        conv_cache: [en_cache, de_cache], (2, B, C, 8(kT-1), F) = (2, 1, 16, 16, 33)
        """
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)  # sfe-lite (B, 3, T, 129)

        # streaming change
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(
            feat, conv_cache[0], tra_cache[0]
        )

        feat, tcn_cache[0] = self.gtcn1(feat, tcn_cache[0])  # (B,16,T,33)
        feat, tcn_cache[1] = self.gtcn2(feat, tcn_cache[1])  # (B,16,T,33)

        # streaming change
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(
            feat, en_outs, conv_cache[1], tra_cache[1]
        )

        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh, conv_cache, tra_cache, tcn_cache


if __name__ == "__main__":
    # load non-streaming model state dict and convert to streaming
    device = torch.device("cpu")
    model = GTCRNMicro().to(device).eval()
    model.load_state_dict(
        torch.load("./gtcrn_micro/ckpts/best_model_dns3.tar", map_location=device)[
            "model"
        ]
    )
    stream_model = StreamGTCRNMicro().to(device).eval()
    convert_to_stream(stream_model, model)

    # offline inference
    print("\nOffline inference")
    x = torch.from_numpy(
        sf.read(
            "./gtcrn_micro/data/DNS3/noisy_blind_testset_v3_challenge_withSNR_16k/ms_realrec_emotional_female_SNR_17.74dB_headset_A2AHXGFXPG6ZSR_Water_far_Laughter_12.wav",
            dtype="float32",
        )[0]
    )
    x = torch.stft(
        x, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
    )[None]

    # adjusted for torch update with return_complex=False
    with torch.no_grad():
        y = model(x)

    # for time domain comparison
    enhanced = torch.view_as_complex(y.contiguous())
    enhanced = torch.istft(
        enhanced, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False
    )
    # enhanced = enhanced.squeeze(0).cpu().numpy()
    # # sf.write("./gtcrn_micro/streaming/sample_wavs/enh.wav", y.squeeze(), 16000)
    # sf.write("./gtcrn_micro/streaming/sample_wavs/enh.wav", enhanced, 16000)

    # --------------
    # streaming inference
    print("\nStreaming inference")
    # conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
    conv_cache = torch.zeros(2, 1, 16, 6, 33).to(device)
    tra_cache = torch.zeros(2, 3, 1, 8, 2).to(device)
    tcn_cache = [
        [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
        [torch.zeros(1, 16, 2 * d, 33, device=device) for d in [1, 2, 4, 8]],
    ]
    ys = []
    times = []
    for i in tqdm(range(x.shape[2])):
        xi = x[:, :, i : i + 1]
        tic = time.perf_counter()
        with torch.no_grad():
            yi, conv_cache, tra_cache, tcn_cache = stream_model(
                xi, conv_cache, tra_cache, tcn_cache
            )
        toc = time.perf_counter()
        times.append((toc - tic) * 1000)
        ys.append(yi)
    ys = torch.cat(ys, dim=2)

    enhanced_stream = torch.view_as_complex(ys.contiguous())
    enhanced_stream = torch.istft(
        enhanced_stream,
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5),
        return_complex=False,
    )
    # enhanced_stream = enhanced_stream.squeeze(0).cpu().numpy()
    # sf.write(
    #     "./gtcrn_micro/streaming/sample_wavs/enh.wav", enhanced_stream.squeeze(), 16000
    # )
    print(
        ">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(
            sum(times) / len(times), max(times), min(times)
        )
    )
    print(">>> Streaming error, FREQ domain:", np.abs(y - ys).max())
    print(">>> Streaming error, TIME domain:", np.abs(enhanced - enhanced_stream).max())

    # --------------
