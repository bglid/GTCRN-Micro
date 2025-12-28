# Original authors: Xiaobin Rong, Xiaohuai Le
# Source: GTCRN: https://github.com/Xiaobin-Rong/gtcrn

from typing import Tuple, Union

import torch
import torch.nn as nn


class StreamConv1d(nn.Module):
    """Causal streaming 1d convolution over time domain. Does forward pass with ring buffer style, but uses slicing instead of pointers.

    Attributes:
        Conv1d: 1d convolution over time domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super(StreamConv1d, self).__init__(*args, **kwargs)

        assert padding == 0, "Padding must be 0 to keep it causal!"

        self.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x, cache):
        """Forward pass for streaming inference.

        Args:
            x : [bs, C, T_size] - new k sample
            cache : [bs, C, T_size-1] - k-1 input samples from last step

        Returns: Conv output at time k given cache, output cache (k-1) inputs for next step
        """
        print(f"\nx shape: {x.shape}")
        print(f"\ncache shape: {cache.shape}")
        input = torch.cat([cache, x], dim=-1)
        output = self.Conv1d(input)
        out_cache = input[..., 1:]
        return output, out_cache


class StreamConv2d(nn.Module):
    """Causal streaming 2d convolution over time and frequency domain. Does forward pass with ring buffer style, but uses slicing instead of pointers.

    Attributes:
        Conv2d: 2d convolution over time domain and frequency domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # setting causual padding for both time and frequency domain
        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError("Invalid padding size!!")

        # make sure time is still causal
        assert self.T_pad == 0, "Time padding must be 0 to keep it causal!"

        self.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x, cache):
        """Forward pass for streaming inference. Over time and frequency domain (2d).

        Args:
            x : [bs, C, T_size=1, F] - new k sample with frequency dim
            cache : [bs, C, T_size-1, F] - k-1 input samples from last step with frequency dim

        Returns: Conv output at time k given cache, output cache (k-1) inputs for next step. Includes frequency domain info.
        """
        input = torch.cat([cache, x], dim=2)
        output = self.Conv2d(input)
        out_cache = input[:, :, 1:]
        return output, out_cache


class StreamConvTranspose2d(nn.Module):
    """Reimplements ConvTranspose2d using Conv2d, manual upsampling, padding, and flipped weights.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of Output channels
        ConvTranspose2d: Recreated ConvTranspose2d with basic conv2d and manual upsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],  # [T_size, F_size]
        stride: Union[int, Tuple[int, int]] = 1,  # [T_stride, F_stride], T_stride == 1
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(kernel_size) is int:
            self.T_size = kernel_size
            self.F_size = kernel_size
        elif type(kernel_size) in [list, tuple]:
            self.T_size, self.F_size = kernel_size
        else:
            raise ValueError("Invalid kernel size!")

        # ensuring we keep a stride of 1
        if type(stride) is int:
            self.T_stride = stride
            self.F_stride = stride
        elif type(stride) in [list, tuple]:
            self.T_stride, self.F_stride = stride
        else:
            raise ValueError("Invalid Stride!!")

        assert self.T_stride == 1, (
            f"Time stride must be 1 in deconv. Got {self.T_stride} instead"
        )

        # ensuring causal padding
        if type(padding) is int:
            self.T_pad = padding
            self.F_pad = padding
        elif type(padding) in [list, tuple]:
            self.T_pad, self.F_pad = padding
        else:
            raise ValueError("Invalid padding size!")

        assert self.T_pad == 0, f"Padding must be 0 in deconv. Got {self.T_pad} instead"

        if type(dilation) is int:
            self.T_dilation = dilation
            self.F_dilation = dilation
        elif type(dilation) in [list, tuple]:
            self.T_dilation, self.F_dilation = dilation
        else:
            raise ValueError("Invalid dilation size!")

        # Implementing ConvTranspose2d using Conv2d by flipping the weights set in convert_to_stream()
        self.ConvTranspose2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(self.T_stride, 1),  # time stride of 1, no time upsampling
            padding=(self.T_pad, 0),  # Time padding is 0, because causality
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x, cache):
        """Simulates ConvTranspose2d with Upsampling and padding based on F size

        Args:
            x : [bs, C, 1, F]
            cache : [bs, C, T-1, F]

        Returns:
            Upsampled convtranspose2d ouput, output cache for next pass

        """
        input = torch.cat([cache, x], dim=2)
        out_cache = input[:, :, 1:]
        bs, C, T, F = input.shape

        # emulating upsampling, adding F_stride-1 0s
        if self.F_stride > 1:
            # [bs,C,T,F] -> [bs,C,T,F,1] -> [bs,C,T,F,F_stride] -> [bs,C,T,F_out]
            input = torch.cat(
                [
                    input[:, :, :, :, None],
                    torch.zeros([bs, C, T, F, self.F_stride - 1]),
                ],
                dim=-1,
            ).reshape([bs, C, T, -1])

            left_pad = self.F_stride - 1
            if self.F_size > 1:
                if left_pad <= self.F_size - 1:
                    input = torch.nn.functional.pad(
                        input,
                        pad=[
                            (self.F_size - 1) * self.F_dilation - self.F_pad,
                            (self.F_size - 1) * self.F_dilation - self.F_pad - left_pad,
                            0,
                            0,
                        ],
                    )
                else:
                    raise (NotImplementedError)
            else:
                raise (NotImplementedError)
        else:  # F_stride = 1, no upsampling, just padding
            input = torch.nn.functional.pad(
                input,
                pad=[
                    (self.F_size - 1) * self.F_dilation - self.F_pad,
                    (self.F_size - 1) * self.F_dilation - self.F_pad,
                ],
            )

        output = self.ConvTranspose2d(input)
        return output, out_cache
