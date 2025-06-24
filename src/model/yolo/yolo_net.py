
import torch.nn as nn
import torch.nn.functional as F

from model.yolo.backbone import ResNetBackbone
from model.yolo.neck import FPN
from model.yolo.head import DetectionHead
import torch
# ---------------------------------------------------------------------------
# MODEL: YOLONet
# ---------------------------------------------------------------------------
class YOLONet(nn.Module):
    def __init__(self, num_classes=2, num_anchors=3):
        super(YOLONet, self).__init__()

        self.backbone = ResNetBackbone()
        self.neck = FPN(
            in_channels=[1024, 512, 256],
            out_channels=256
        )

        self.head_13x13 = DetectionHead(256, num_classes, num_anchors)
        self.head_26x26 = DetectionHead(256, num_classes, num_anchors)
        self.head_52x52 = DetectionHead(256, num_classes, num_anchors)

    def forward(self, x):
        c3, c2, c1 = self.backbone(x)
        f3, f2, f1 = self.neck(c3, c2, c1)
        out_13 = self.head_13x13(f3)
        out_26 = self.head_26x26(f2)
        out_52 = self.head_52x52(f1)
        return out_13, out_26, out_52

    

