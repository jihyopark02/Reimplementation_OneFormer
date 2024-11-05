import torch
import torch.nn as nn
import torchvision.models as models

class BackboneWithMultiScaleFeatures(nn.Module):
    def __init__(self):
        super(BackboneWithMultiScaleFeatures, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Output shape: (N, 256, H/4, W/4)
        self.layer2 = nn.Sequential(*list(resnet.children())[5])   # Output shape: (N, 512, H/8, W/8)
        self.layer3 = nn.Sequential(*list(resnet.children())[6])   # Output shape: (N, 1024, H/16, W/16)
        self.layer4 = nn.Sequential(*list(resnet.children())[7])   # Output shape: (N, 2048, H/32, W/32)

    def forward(self, x):
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features