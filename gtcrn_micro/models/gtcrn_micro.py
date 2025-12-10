"""
GTCRN-Micro: MCU-focused rebuild of GTCRN
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


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


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""

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

        # self.sfe = SFE(kernel_size=3, stride=1)

        # if removing SFE, remove * 3 because we don't have a kernel of size 3
        self.point_conv1 = conv_module(in_channels // 2, hidden_channels, 1)  # no SFE
        # self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        if use_deconv:
            self.depth_conv = conv_module(
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
            self.depth_conv = conv_module(
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

        # self.tra = TRA(in_channels // 2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, "b c g t f -> b (c g) t f")  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        # h1 = self.tra(h1)

        x = self.shuffle(h1, x2)

        return x


class TCN(nn.Module):
    """
    Temporal Convolutional Block
    2D convolution here
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(
                    3,  # no SFE
                    # 3 * 3,  # SFE
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
                    groups=1,  # switched for conversion
                    # groups=2,
                    use_deconv=False,
                    is_last=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    use_deconv=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    # dilation=(2, 1), # switched for LiteRT inference
                    use_deconv=False,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 1),
                    dilation=(1, 1),
                    # dilation=(5, 1), # switched for LiteRT inference
                    use_deconv=False,
                ),
            ]
        )

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 1, 1),
                    dilation=(1, 1),
                    # padding=(2 * 5, 1), # switched for LiteRT inference
                    # dilation=(5, 1),
                    use_deconv=True,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 1, 1),
                    dilation=(1, 1),
                    # switched for LiteRT inference
                    # dilation=(2, 1),
                    use_deconv=True,
                ),
                GTConvBlock(
                    16,
                    16,
                    (3, 3),
                    stride=(1, 1),
                    padding=(2 * 1, 1),
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
                    # groups=2,
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

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            # print(x.shape)
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRNMicro(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
    ):
        super().__init__()
        self.erb = ERB(65, 64)
        # self.sfe = SFE(3, 1)

        self.encoder = Encoder()

        self.dptcn1 = GTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)
        self.dptcn2 = GTCN(channels=16, n_layers=4, kernel_size=3, dilation=2)

        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, spec):
        """
        spec: (B, F, T, 2)
        """
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)

        feat, en_outs = self.encoder(feat)

        feat = self.dptcn1(feat)  # (B,16,T,33)
        feat = self.dptcn2(feat)  # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)

        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh


if __name__ == "__main__":
    model = GTCRNMicro().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(
        model, (257, 63, 2), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print(flops, params)

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

    print((y1[: 16000 - 256 * 2] - y2[: 16000 - 256 * 2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
