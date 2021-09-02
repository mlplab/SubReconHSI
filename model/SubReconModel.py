# coding: utf-8


import torch
from .layers import Base_Module


class RGBHSCNN(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, **kwargs) -> None:
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        if input_ch == 0:
            self.input_conv = torch.nn.Identity()
        else:
            self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.input_activation = self.activations[activation]()
        self.feature_layers = torch.nn.ModuleDict({f'RGB_{i}': torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
                                                   for i in range(layer_num)})
        self.activation_layer = torch.nn.ModuleDict({f'RGB_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_activation(self.input_conv(x))
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        x = self.output_conv(x)
        return x


class HSIHSCNN(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num: int=3, **kwargs) -> None:
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.input_activation = self.activations[activation]()
        self.feature_layers = torch.nn.ModuleDict({f'HSI_{i}': torch.nn.Conv2d(feature_num, feature_num, 3, 1, 1)
                                                   for i in range(layer_num)})
        self.activation_layer = torch.nn.ModuleDict({f'HSI_act_{i}': self.activations[activation]()
                                                     for i in range(layer_num)})
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_activation(self.input_conv(x))
        for layer, activation in zip(self.feature_layers.values(), self.activation_layer.values()):
            x = activation(layer(x))
        x = self.output_conv(x)
        return x


class SpectralFusion(Base_Module):

    def __init__(self, input_hsi_ch: int, input_rgb_ch: int, output_hsi_ch: int,
                 output_rgb_ch: int, *args, rgb_feature: int=64, hsi_feature: int=64,
                 fusion_feature: int=64, layer_num: int=3, **kwargs) -> None:
        super().__init__()
        self.input_rgb_ch = input_rgb_ch
        self.output_rgb_ch = output_rgb_ch
        self.layer_num = layer_num
        self.rgb_layer = RGBHSCNN(input_rgb_ch, output_rgb_ch, feature_num=rgb_feature,
                                  layer_num=layer_num)
        self.hsi_layer = HSIHSCNN(input_hsi_ch, output_hsi_ch, feature_num=hsi_feature,
                                  layer_num=layer_num)
        self.fusion_conv = torch.nn.ModuleDict({f'Fusion_{i}': torch.nn.Conv2d(rgb_feature + hsi_feature, fusion_feature, 1, 1, 0)
                                                for i in range(layer_num)})

    def forward(self, rgb: torch.Tensor, hsi: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        hsi_x = self.hsi_layer.input_activation(self.hsi_layer.input_conv(hsi))
        if self.input_rgb_ch >= 1:
            rgb_x = self.rgb_layer.input_activation(self.rgb_layer.input_conv(rgb))
        else:
            rgb_x = hsi_x
        for i in range(self.layer_num):
            rgb_x = self.rgb_layer.activation_layer[f'RGB_act_{i}'](self.rgb_layer.feature_layers[f'RGB_{i}'](rgb_x))
            fusion_feature = torch.cat((rgb_x, hsi_x), dim=1)
            hsi_x = self.fusion_conv[f'Fusion_{i}'](fusion_feature)
            hsi_x = self.hsi_layer.activation_layer[f'HSI_act_{i}'](self.hsi_layer.feature_layers[f'HSI_{i}'](hsi_x))
        output_hsi = self.hsi_layer.output_conv(hsi_x)
        if self.output_rgb_ch >= 1:
            output_rgb = self.rgb_layer.output_conv(rgb_x)
            return output_rgb, output_hsi
        else:
            return output_hsi
