from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

@dataclass
class BackboneOutput:
    cls_feature: torch.Tensor = field(default=None)
    feature_maps: list[torch.Tensor] = field(default_factory=list)

class Backbone(nn.Module):
    def forward(self, img: torch.Tensor, *args, **kwargs) -> BackboneOutput:
        raise NotImplementedError

@dataclass
class DecoderOutput:
    # low->high resolution
    feature_maps: list[torch.Tensor]

class Decoder(nn.Module):
    def forward(self, backbone_feature_maps: list[torch.Tensor], x_in) -> DecoderOutput:
        raise not NotImplementedError
