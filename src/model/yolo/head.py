import torch
import torch.nn as nn

# ---
## 3. Head Class
# ---
class DetectionHead(nn.Module):
    """
    Simulates a Detection Head.
    Task: Predict bounding box, confidence score, and class probabilities.
    """
    def __init__(self, in_channels, num_classes, num_anchors):
        super(DetectionHead, self).__init__()
        # Number of output channels predicted for each grid cell:
        # num_anchors * (4 (box_coords) + 1 (confidence) + num_classes)
        self.output_channels = num_anchors * (4 + 1 + num_classes)

        # A simple convolution layer to produce predictions
        self.conv_predict = nn.Conv2d(in_channels, self.output_channels, kernel_size=1)

    def forward(self, x):
        # x is the feature map from Neck
        # Example: (batch_size, 256, 13, 13)
        prediction = self.conv_predict(x)
        # reshape to handle anchor boxes if needed
        # prediction_reshaped = prediction.view(
        #     x.size(0), self.num_anchors, (4 + 1 + self.num_classes), x.size(2), x.size(3)
        # ).permute(0, 3, 4, 1, 2).contiguous() # Permute for easier processing
        return prediction # For simplicity, just return the raw output tensor