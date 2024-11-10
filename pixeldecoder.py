import torch.nn as nn
import torch.nn.functional as F

class PixelDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=256):

        super(PixelDecoder, self).__init__()
        self.output_channels = output_channels

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, output_channels, kernel_size=1)
            for in_channels in input_channels
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in input_channels
        ])

    def forward(self, features):
        multi_scale_features = []
        x = self.lateral_convs[-1](features[-1]) 
        multi_scale_features.append(self.output_convs[-1](x))  

        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")  
            lateral = self.lateral_convs[i](features[i])           
            x = x + lateral                                       
            multi_scale_features.append(self.output_convs[i](x))  

        return multi_scale_features[::-1]
