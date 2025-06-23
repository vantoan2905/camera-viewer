from torch import nn
from torch.nn import functional as F
# ---------------------------------------------------------------------------
# NECK: FPN
# ---------------------------------------------------------------------------
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral1 = nn.Conv2d(in_channels[2], out_channels, 1)

        self.output3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.output1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c2, c1):
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')

        o3 = self.output3(p3)
        o2 = self.output2(p2)
        o1 = self.output1(p1)
        return o3, o2, o1
