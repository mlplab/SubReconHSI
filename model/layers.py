# coding: utf-8


import torch
import torchvision
import numpy  as np


# ########################## Activation Function ##########################


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.functional.softplus(x))


class ReLU(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(ReLU, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class Leaky(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Leaky, self).__init__()
        self.alpha = kwargs.get('alpha', .02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0., x * self.alpha, x)


class Swish(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Swish, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class Mish(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(Mish, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


# ########################## Loss Function ##########################


class RMSELoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(x, y))


class SAMLoss(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        xy = torch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class MSE_SAMLoss(torch.nn.Module):

    def __init__(self, alpha: float=.5, beta: float=.5, mse_ratio: float=1., sam_ratio: float=.01) -> None:
        super(MSE_SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss()
        self.sam_loss = SAMLoss()
        self.mse_ratio = mse_ratio
        self.sam_ratio = sam_ratio

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse_ratio * self.mse_loss(x, y) + self.beta * self.sam_ratio * self.sam_loss(x, y)


class FusionLoss(torch.nn.Module):

    def __init__(self, rgb_base_fn: torch.nn.Module=torch.nn.MSELoss,
                 hsi_base_fn: torch.nn.Module=torch.nn.MSELoss, **kwargs) -> None:
        super().__init__()
        self.rgb_fn = rgb_base_fn()
        self.hsi_fn = hsi_base_fn()

    def forward(self, output: tuple, label: tuple) -> torch.Tensor:
        rgb_x, hsi_x = output
        if isinstance(label, (dict)):
            rgb_y, hsi_y = label['rgb'], label['hsi']
        else:
            rgb_y, hsi_y = label
        return .5 * self.rgb_fn(rgb_x, rgb_y) + .5 * self.hsi_fn(hsi_x, hsi_y)


# ########################## Loss Function ##########################


class Base_Module(torch.nn.Module):

    def __init__(self) -> None:
        super(Base_Module, self).__init__()
        self.activations = {'relu': ReLU, 'leaky': Leaky, 'mish': Mish, 'swish': Swish,
                            'none': torch.nn.Identity}


class DW_PT_Conv(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, kernel_size: int, activation: str='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activations[activation]()
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.depth(x)
        x = self.activation(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class HSI_prior_block(Base_Module):

    def __init__(self, input_ch: int, output_ch: int, feature: int=64,
                 activation: str='relu') -> None:
        super(HSI_prior_block, self).__init__()
        self.activation = self.activations[activation]()
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        h = self.spatial_1(x)
        h = self.activation(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class My_HSI_network(Base_Module):

    def __init__(self, input_ch: int, output_ch: int,
                 feature: int=64, activation: str='relu'):
        super(My_HSI_network, self).__init__()
        self.activation = self.activation[activation]()
        self.spatial_1 = DW_PT_Conv(input_ch, feature, 3, activation=None)
        self.spatial_2 = DW_PT_Conv(feature, output_ch, 3, activation=None)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        h = self.spatial_1(x)
        h = self.activation(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class Conv2d(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 kernel_size: int=3, stride: int=1, **kwargs) -> None:

        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size,
                                    stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Depthwise(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 kernel_size: int=3, stride: int=1, **kwargs) -> None:

        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=input_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ########################## DeformableConv2d  ##########################
# https://github.com/ChunhuanLin/deform_conv_pytorch


class DeformableConv2d(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, **kwargs) -> None:

        super().__init__()
        kernel_size = 3
        stride = 1
        self.padding = kernel_size // 2
        self.offset_conv = torch.nn.Conv2d(input_ch, 2 * kernel_size * kernel_size,
                kernel_size, stride, self.padding)
        self.modulator_conv = torch.nn.Conv2d(input_ch, 1 * kernel_size * kernel_size,
                kernel_size, stride, self.padding)
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride, self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset,
                                         weight=self.conv.weight,
                                         bias=self.conv.bias,
                                         padding=self.padding,
                                         mask=modulator)
        return x


# ########################## Mix and Ghost Conv 2d  ##########################
def split_layer(output_ch: int, chunks: int):
    split = [np.int(np.ceil(output_ch / chunks)) for _ in range(chunks)]
    split[chunks - 1] += output_ch - sum(split)
    return split


class Ghost_Mix(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 kernel_size: int=3, stride: int=1, dw_kernel: int=3,
                 dw_stride: int=1, ratio: int=2, **kwargs) -> None:
        super(Ghost_Mix, self).__init__()
        self.ratio = ratio
        self.mode = kwargs.get('mode', 'mix3').lower()
        chunks = kwargs.get('chunks', ratio - 1)
        self.output_ch = output_ch
        primary_ch = int(np.ceil(output_ch / ratio))
        cheap_ch = primary_ch * (self.ratio - 1)
        self.activation = kwargs.get('activation', 'relu').lower()
        self.primary_conv = torch.nn.Conv2d(input_ch, primary_ch, kernel_size, stride, padding=kernel_size // 2)
        cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                            dw_stride, dw_kernel // 2, groups=primary_ch)
        mix = Mix_SS_Layer(cheap_ch, cheap_ch, chunks=chunks, groups=ratio - 1)
        self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix])
        '''
        if self.mode == 'mix1':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_Conv(cheap_ch, cheap_ch, chunks=chunks)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix])
        elif self.mode == 'mix2':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_Conv(cheap_ch, cheap_ch, chunks=chunks)
            pw = torch.nn.Conv2d(cheap_ch, cheap_ch, 1, 1, 0)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix, pw])
        elif self.mode == 'mix3':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_SS_Layer(cheap_ch, cheap_ch, chunks=chunks, groups=ratio - 1)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix])
        elif self.mode == 'mix4':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_SS_Layer(cheap_ch, cheap_ch, chunks=chunks, groups=ratio - 1)
            pw = torch.nn.Conv2d(cheap_ch, cheap_ch, 1, 1, 0)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix, pw])
        else:
            self.cheap_conv = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                              dw_stride, padding=dw_kernel // 2, groups=primary_ch)
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary_x = self.primary_conv(x)
        cheap_x = self.cheap_conv(primary_x)
        output = torch.cat([primary_x, cheap_x], dim=1)
        return output[:, :self.output_ch, :, :]



class Ghost_normal(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args,
                 kernel_size: int=3, stride: int=1, dw_kernel: int=3,
                 dw_stride: int=1, ratio: int=2, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.mode = kwargs.get('mode', 'mix3').lower()
        chunks = kwargs.get('chunks', ratio - 1)
        self.output_ch = output_ch
        primary_ch = int(np.ceil(output_ch / ratio))
        cheap_ch = primary_ch * (self.ratio - 1)
        self.activation = kwargs.get('activation', 'relu').lower()
        self.primary_conv = torch.nn.Conv2d(input_ch, primary_ch, kernel_size, stride, padding=kernel_size // 2)
        self.cheap_conv = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                          dw_stride, padding=dw_kernel // 2, groups=primary_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary_x = self.primary_conv(x)
        cheap_x = self.cheap_conv(primary_x)
        output = torch.cat([primary_x, cheap_x], dim=1)
        return output[:, :self.output_ch, :, :]


class GroupConv(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, chunks: int,
                 kernel_size: int, *args, stride: int=1, **kwargs):
        super(GroupConv, self).__init__()
        self.chunks = chunks
        self.split_input_ch = split_layer(input_ch, chunks)
        self.split_output_ch = split_layer(output_ch, chunks)

        if chunks == 1:
            self.group_conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2)
        else:
            self.group_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_input_ch[idx],
                                                                     self.split_output_ch[idx],
                                                                     kernel_size, stride=stride,
                                                                     padding=kernel_size // 2) for idx in range(chunks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.chunk(x, self.chunks, dim=1)
            return torch.cat([group_layer(split_x) for group_layer, split_x in zip(self.group_layers, split)], dim=1)


class Mix_Conv(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, chunks: int, stride: int=1, **kwargs) -> None:
        super(Mix_Conv, self).__init__()

        self.chunks = chunks
        self.output_split = split_layer(output_ch, chunks)
        self.input_split = split_layer(input_ch, chunks)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.input_split[idx],
                                                                self.output_split[idx],
                                                                kernel_size=2 * idx + 3,
                                                                stride=stride,
                                                                padding=(2 * idx + 3) // 2,
                                                                groups=self.input_split[idx]) for idx in range(chunks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.chunk(x, self.chunks, dim=1)
        output = torch.cat([conv_layer(split_x) for conv_layer, split_x in zip(self.conv_layers, split)], dim=1)
        return output


class Group_SE(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, chunks: int,
                 kernel_size: int, **kwargs):
        super(Group_SE, self).__init__()
        ratio = kwargs.get('ratio', 2)
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        if ratio == 1:
            feature_num = input_ch
            self.squeeze = torch.nn.Sequential()
        else:
            feature_num = max(1, output_ch // ratio)
            self.squeeze = GroupConv(input_ch, feature_num, chunks, kernel_size, 1, 0)
        self.extention = GroupConv(feature_num, output_ch, chunks, kernel_size, 1, 0)
        self.activation = activations[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = torch.mean(x, [2, 3], keepdim=True)
        squeeze = self.activation(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torch.sigmoid(extention) * x


class Mix_Conv(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, chunks: int, stride: int=1, **kwargs):
        super(Mix_Conv, self).__init__()

        self.chunks = chunks
        self.split_layer = split_layer(output_ch, chunks)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_layer[idx],
                                                                self.split_layer[idx],
                                                                kernel_size=2 * idx + 3,
                                                                stride=stride,
                                                                padding=(2 * idx + 3) // 2,
                                                                groups=self.split_layer[idx]) for idx in range(chunks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.chunk(x, self.chunks, dim=1)
        output = torch.cat([conv_layer(split_x) for conv_layer, split_x in zip(self.conv_layers, split)], dim=1)
        return output


class Mix_SS_Layer(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, chunks: int, *args,
                 feature_num: int=64, group_num: int=1, **kwargs) -> None:
        super(Mix_SS_Layer, self).__init__()
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        se_flag = kwargs.get('se_flag', False)
        self.spatial_conv = GroupConv(input_ch, feature_num, group_num, kernel_size=3, stride=1)
        self.mix_conv = Mix_Conv(feature_num, feature_num, chunks)
        self.se_block = Group_SE(feature_num, feature_num, chunks, kernel_size=1) if se_flag is True else torch.nn.Sequential()
        self.spectral_conv = GroupConv(feature_num, output_ch, group_num, kernel_size=1, stride=1)
        self.shortcut = torch.nn.Sequential()
        self.spatial_activation = activations[activation]()
        self.mix_activation = activations[activation]()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spatial_activation(self.spatial_conv(x))
        h = self.mix_activation(self.mix_conv(h))
        h = self.se_block(h)
        h = self.spectral_conv(h)
        return h + self.shortcut(x)
