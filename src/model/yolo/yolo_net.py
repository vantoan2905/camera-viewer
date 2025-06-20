import torch
import torch.nn as nn
import torch.nn.functional as F

from model.yolo.backbone import Backbone
from model.yolo.neck import Neck
from model.yolo.head import DetectionHead




# ---
## 4. YOLONet Class (Combine the whole model)
# ---
class YOLONet(nn.Module):
    """
    Combines Backbone, Neck, and Detection Heads to form a complete YOLO model.
    """
    def __init__(self, num_classes=20, num_anchors=3):
        super(YOLONet, self).__init__()
        self.backbone = Backbone()

        # Get output channels from Backbone to initialize Neck
        backbone_13x13_channels = self.backbone.output_13x13_channels
        backbone_26x26_channels = self.backbone.output_26x26_channels
        backbone_52x52_channels = self.backbone.output_52x52_channels

        self.neck = Neck(backbone_13x13_channels,
                         backbone_26x26_channels,
                         backbone_52x52_channels)

        # Prediction heads for each feature map size
        # Assume Neck outputs 256 channels for each scale
        self.head_13x13 = DetectionHead(256, num_classes, num_anchors) # Predict large objects
        self.head_26x26 = DetectionHead(256, num_classes, num_anchors) # Predict medium objects
        self.head_52x52 = DetectionHead(256, num_classes, num_anchors) # Predict small objects

    def forward(self, x):
        # 1. Pass through Backbone
        feature_13x13_bb, feature_26x26_bb, feature_52x52_bb = self.backbone(x)
        # print(f"Backbone 13x13 feature shape: {feature_13x13_bb.shape}")
        # print(f"Backbone 26x26 feature shape: {feature_26x26_bb.shape}")
        # print(f"Backbone 52x52 feature shape: {feature_52x52_bb.shape}\n")

        # 2. Pass through Neck
        neck_out_13x13, neck_out_26x26, neck_out_52x52 = self.neck(
            feature_13x13_bb, feature_26x26_bb, feature_52x52_bb
        )
        # print(f"Neck 13x13 output shape: {neck_out_13x13.shape}")
        # print(f"Neck 26x26 output shape: {neck_out_26x26.shape}")
        # print(f"Neck 52x52 output shape: {neck_out_52x52.shape}\n")

        # 3. Pass through Heads to produce predictions
        prediction_13x13 = self.head_13x13(neck_out_13x13)
        prediction_26x26 = self.head_26x26(neck_out_26x26)
        prediction_52x52 = self.head_52x52(neck_out_52x52)

        # print(f"Head 13x13 prediction shape: {prediction_13x13.shape} (Large objects)")
        # print(f"Head 26x26 prediction shape: {prediction_26x26.shape} (Medium objects)")
        # print(f"Head 52x52 prediction shape: {prediction_52x52.shape} (Small objects)\n")

        return prediction_13x13, prediction_26x26, prediction_52x52
