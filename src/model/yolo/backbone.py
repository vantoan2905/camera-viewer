import torch
import torch.nn as nn
import torch.nn.functional as F


# ---
## 1. Backbone Class
# ---
class Backbone(nn.Module):
    """
    Simulates the Backbone part of the YOLO model.
    Task: Extract features from the input image and produce feature maps at different sizes.
    """
    def __init__(self):
        super(Backbone, self).__init__()
        # For simplicity, we use convolution and pooling layers.
        # In practice, the Backbone can be Darknet, ResNet, CSPNet, etc.

        # Layer 1: Downsample to 1/2 (e.g., 416x416 -> 208x208)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 208x208

        # Layer 2: Downsample to 1/4 (e.g., 208x208 -> 104x104)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 104x104

        # Layer 3: Downsample to 1/8 (e.g., 104x104 -> 52x52)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 52x52
        self.output_52x52_channels = 128 # Output channels of 52x52 feature map

        # Layer 4: Downsample to 1/16 (e.g., 52x52 -> 26x26)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 26x26
        self.output_26x26_channels = 256 # Output channels of 26x26 feature map

        # Layer 5: Downsample to 1/32 (e.g., 26x26 -> 13x13)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 13x13
        self.output_13x13_channels = 512 # Output channels of 13x13 feature map

    def forward(self, x):
        # Input image has shape (batch_size, 3, H, W)
        # In this example, assume H=W=416

        x = F.relu(self.conv1(x))
        x = self.pool1(x) # Output: (batch_size, 32, 208, 208)

        x = F.relu(self.conv2(x))
        x = self.pool2(x) # Output: (batch_size, 64, 104, 104)

        x = F.relu(self.conv3(x))
        feature_52x52 = self.pool3(x) # Output: (batch_size, 128, 52, 52)

        x = F.relu(self.conv4(feature_52x52))
        feature_26x26 = self.pool4(x) # Output: (batch_size, 256, 26, 26)

        x = F.relu(self.conv5(feature_26x26))
        feature_13x13 = self.pool5(x) # Output: (batch_size, 512, 13, 13)

        # Return feature maps at different sizes
        return feature_13x13, feature_26x26, feature_52x52