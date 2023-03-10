from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution, get_output_padding, get_padding
from monai.networks.layers import Act, DropPath, Norm, get_act_layer

def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    stride: Sequence[int] | int = 1,
    groups: int = 1,
    dropout: tuple | str | float | None = None,
    norm: tuple | str | None = Norm.INSTANCE,
    act: tuple | str | None = Act.LEAKYRELU,
    adn_ordering: str = "DNA",
    bias: bool = False,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        _spatial_dims := 3,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        adn_ordering=adn_ordering,
        act=act,
        norm=norm,
        dropout=dropout,
        groups=groups,
        bias=bias,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

class ResBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
        drop_path: float = .0,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            dropout=dropout,
            norm=norm,
            act=act,
        )
        self.conv2 = get_conv_layer(
            out_channels,
            out_channels,
            kernel_size,
            dropout=dropout,
            norm=norm,
            act=None,
        )
        if in_channels != out_channels:
            self.res = get_conv_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                dropout=dropout,
                norm=norm,
                act=None,
            )
        else:
            self.res = nn.Identity()
        self.act2 = get_act_layer(act)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        res = self.res(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop_path(x) + res
        x = self.act2(x)
        return x

class ResLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        dropout: tuple | str | float | None = None,
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
        drop_paths: float | Sequence[float] = 0.
    ):
        super().__init__()
        if isinstance(drop_paths, float):
            drop_paths = [drop_paths] * num_blocks
        assert len(drop_paths) == num_blocks
        self.blocks = nn.Sequential(
            ResBasicBlock(in_channels, out_channels, kernel_size, dropout, norm, act, drop_paths[0]),
            *[
                ResBasicBlock(out_channels, out_channels, kernel_size, dropout, norm, act, drop_path)
                for drop_path in drop_paths[1:]
            ],
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)
