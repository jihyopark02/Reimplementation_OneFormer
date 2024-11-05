import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelDecoder(nn.Module):
    def __init__(self, input_channels, output_channels=256):
        """
        Args:
            input_channels (list of int): Number of input channels for each feature map from the backbone.
            output_channels (int): Number of output channels for each feature map after decoding.
        """
        super(PixelDecoder, self).__init__()
        self.output_channels = output_channels

        # Define lateral convolutions for each input feature map
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, output_channels, kernel_size=1)
            for in_channels in input_channels
        ])

        # Define output convolutions after combining features
        self.output_convs = nn.ModuleList([
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in input_channels
        ])

    def forward(self, features):
        """
        Args:
            features (list[torch.Tensor]): List of feature maps from the backbone, ordered from high to low resolution.
        Returns:
            multi_scale_features (list[torch.Tensor]): List of feature maps with consistent channels.
        """
        # Start from the last feature map, which is the smallest spatial resolution
        multi_scale_features = []
        x = self.lateral_convs[-1](features[-1])  # Apply lateral conv on the smallest feature map
        multi_scale_features.append(self.output_convs[-1](x))  # Store the final output for this scale

        # Process each feature map in top-down order
        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # Upsample the current feature map
            lateral = self.lateral_convs[i](features[i])           # Apply lateral conv to the next feature map
            x = x + lateral                                        # Add upsampled feature with lateral connection
            multi_scale_features.append(self.output_convs[i](x))   # Apply output conv and store result

        # Reverse the output list to match increasing spatial resolution order
        return multi_scale_features[::-1]