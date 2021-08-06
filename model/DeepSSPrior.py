# coding: utf-8


import torch
from .layers import swish, mish, HSI_prior_block


class DeepSSPrior(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, feature: int=64, 
                 block_num: int=9, activation: str='relu', **kwargs) -> None:
        super(DeepSSPrior, self).__init__()
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        self.hsi_block = torch.nn.ModuleDict({f'HSI_Block_{i:02d}': HSI_prior_block(output_ch, output_ch, feature=feature, activation=activation) for i in range(block_num)})
        self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start_conv(x)
        x_in = x
        for (layer_name, hsi_block) in self.hsi_block.items():
            x_hsi = hsi_block(x)
            x_res = self.residual_block(x)
            x = x_res + x_hsi + x_in
        return self.output_conv(x)
