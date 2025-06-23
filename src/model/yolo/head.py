import torch
import torch.nn as nn
from model.yolo.backbone import ResNetBackbone
from model.yolo.neck import FPN 
# ---------------------------------------------------------------------------
# HEAD: Detection Head
# ---------------------------------------------------------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super(DetectionHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            num_anchors * (4 + 1 + num_classes),
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)