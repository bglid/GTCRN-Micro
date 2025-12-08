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

        # Streaming!
        # self.depth_conv = stream_conv_module(
        #     hidden_channels,
        #     hidden_channels,
        #     kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     # groups=hidden_channels,
        #     groups=1,  # fixing for conversion
        # )
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

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)."""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        # adjusting for streaming
        # x = rearrange(x, "b c g t f -> b (c g) t f")  # (B,2C,T,F)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

        return x

    def forward(self, x, conv_cache):
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

        x = self.shuffle(h1, x2)

        return x, conv_cache


# todo: change to TCN once everything else works
class TCN(nn.Module):
    """
    Streaming Temporal Convolutional Block.
    """

    def __init__(self, channels, kernel_size=3, dilation=1) -> None:
        super().__init__()
        # padding setup
        self.pad = dilation * (kernel_size - 1)

        # conv1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU()

        # dep temporal conv
        self.conv2 = nn.Conv2d(
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

    def forward(self, x):
        """
        x: (batch, seq length, input size, freq bins)
        """

        # we are going to use a residual connection within this block
        residual = x

        # first conv pass
        y1 = self.act1(self.bn1(self.conv1(x)))

        # doing padding
        y1 = nn.functional.pad(y1, [0, 0, self.pad, 0])
        # Conv2 pass
        y2 = self.act2(self.bn2(self.conv2(y1)))

        # conv 3 but doing the activation with the resid
        y3 = self.bn3(self.conv3(y2))

        res = y3 + residual
        return self.act3(res)


class GTCN(nn.Module):
    """ """

    def __init__(self, channels, n_layers=4, kernel_size=3, dilation=2) -> None:
        super().__init__()
        # trying to stack into blocks to recreate dp
        blocks = []
        self.dilation = 1  # going to increase this
        for i in range(n_layers):
            blocks.append(
                TCN(channels=channels, kernel_size=kernel_size, dilation=dilation)
            )
            self.dilation *= dilation  # increases each block

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        x: (B, C, T, F).
        """
        for tcn in self.blocks:
            x = tcn(x)

        return x


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

    def forward(self, x, conv_cache):
        en_outs = []
        # just passing as normal through first two conv blocks
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)

        # streaming forward pass for Streaming blocks
        # NOTE: due to dilation quant restrictions, need to make smaller dilation caches
        x, conv_cache[:, :, :2, :] = self.en_convs[2](x, conv_cache[:, :, :2, :])
        en_outs.append(x)

        x, conv_cache[:, :, 2:4, :] = self.en_convs[3](x, conv_cache[:, :, 2:4, :])
        en_outs.append(x)

        x, conv_cache[:, :, 4:6, :] = self.en_convs[4](x, conv_cache[:, :, 4:6, :])
        en_outs.append(x)

        return x, en_outs, conv_cache


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

    def forward(self, x, en_outs, conv_cache):
        # decoding the cache backwards
        x, conv_cache[:, :, 4:6, :] = self.de_convs[0](
            x + en_outs[4], conv_cache[:, :, 4:6, :]
        )

        x, conv_cache[:, :, 2:4, :] = self.de_convs[1](
            x + en_outs[3], conv_cache[:, :, 2:4, :]
        )

        x, conv_cache[:, :, :2, :] = self.de_convs[2](
            x + en_outs[2], conv_cache[:, :, :2, :]
        )

        # iter as normal through last two conv blocks
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4 - i])
        return x, conv_cache


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

        self.encoder = StreamEncoder()

        self.dptcn1 = GTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)
        self.dptcn2 = GTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)

        self.decoder = StreamDecoder()

        self.mask = Mask()

    def forward(self, spec, conv_cache):
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

        # streaming change
        # print(f"\nDEBUG\n-------\nConv cache:{conv_cache[0]}\n--------")
        feat, en_outs, conv_cache[0] = self.encoder(feat, conv_cache[0])

        feat = self.dptcn1(feat)  # (B,16,T,33)
        feat = self.dptcn2(feat)  # (B,16,T,33)

        # streaming change
        m_feat, conv_cache[1] = self.decoder(feat, en_outs, conv_cache[1])

        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh, conv_cache


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
    ys = []
    times = []
    for i in tqdm(range(x.shape[2])):
        xi = x[:, :, i : i + 1]
        tic = time.perf_counter()
        with torch.no_grad():
            yi, conv_cache = stream_model(xi, conv_cache)
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
    print(
        ">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(
            sum(times) / len(times), max(times), min(times)
        )
    )
    print(">>> Streaming error, FREQ domain:", np.abs(y - ys).max())
    print(">>> Streaming error, TIME domain:", np.abs(enhanced - enhanced_stream).max())

    # --------------
