import torch.nn as nn
import torchvision.models as models

class BackboneWithMultiScaleFeatures(nn.Module):
    def __init__(self):
        super(BackboneWithMultiScaleFeatures, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5])
        self.layer3 = nn.Sequential(*list(resnet.children())[6])
        self.layer4 = nn.Sequential(*list(resnet.children())[7])

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