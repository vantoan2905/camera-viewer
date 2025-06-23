from torch import nn


from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = resnet50(pretrained=True)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

        self.output_52x52_channels = 256  # giả sử lấy từ stage1
        self.output_26x26_channels = 512  # từ stage2
        self.output_13x13_channels = 1024  # từ stage3

    def forward(self, x):
        c1 = self.stage1(x)   # ~52x52
        c2 = self.stage2(c1)  # ~26x26
        c3 = self.stage3(c2)  # ~13x13
        return c3, c2, c1
