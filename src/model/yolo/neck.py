import torch
import torch.nn as nn
import torch.nn.functional as F
# ---
## 2. Neck Class
# ---
class Neck(nn.Module):
    """
    Simulates the Neck part of the YOLO model (e.g., FPN/PANet).
    Task: Combine semantic information from deep layers with spatial information from shallow layers.
    """
    def __init__(self, in_channels_13x13, in_channels_26x26, in_channels_52x52):
        super(Neck, self).__init__()
        # For simplicity, we only use simple convolution layers to "combine"
        # In practice, architectures like FPN/PANet are much more complex.

        # Convolution layers to process feature maps from Backbone
        # (Here we assume the output channels after Neck is 256 for all scales)
        self.conv_13x13 = nn.Conv2d(in_channels_13x13, 256, kernel_size=1)
        self.conv_26x26 = nn.Conv2d(in_channels_26x26, 256, kernel_size=1)
        self.conv_52x52 = nn.Conv2d(in_channels_52x52, 256, kernel_size=1)

        # UpSampling layer to combine feature maps (simulate FPN top-down path)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, feature_13x13, feature_26x26, feature_52x52):
        # feature_13x13: (batch_size, 512, 13, 13)
        # feature_26x26: (batch_size, 256, 26, 26)
        # feature_52x52: (batch_size, 128, 52, 52)

        # Process feature maps from Backbone
        neck_out_13x13 = self.conv_13x13(feature_13x13)
        neck_out_26x26 = self.conv_26x26(feature_26x26)
        neck_out_52x52 = self.conv_52x52(feature_52x52)

        # FPN-like connection: Upsample and add
        # Combine 13x13 with 26x26
        upsampled_13x13 = self.upsample(neck_out_13x13) # (batch_size, 256, 26, 26)
        # Ensure sizes match before adding (may need cropping if not perfectly matched)
        combined_26x26 = upsampled_13x13 + neck_out_26x26 # (batch_size, 256, 26, 26)

        # Combine combined_26x26 with 52x52
        upsampled_26x26 = self.upsample(combined_26x26) # (batch_size, 256, 52, 52)
        combined_52x52 = upsampled_26x26 + neck_out_52x52 # (batch_size, 256, 52, 52)

        # Return feature maps that have been combined and enriched
        return neck_out_13x13, combined_26x26, combined_52x52