# coding: utf-8


import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from colour.colorimetry import transformations
import torch
import torchvision
from torchinfo import summary
from .layers import ReLU, Leaky, Swish, Mish


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


class HSCNN(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature: int=64,
                 block_num: int=9, **kwargs) -> None:
        super(HSCNN, self).__init__()
        self.output_ch = output_ch
        activation = kwargs.get('activation', 'leaky')
        mode = kwargs.get('mode', 'add')
        self.residual = kwargs.get('res', True)
        self.feature_num = feature
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_activation = activations[activation]()
        self.patch_extraction = torch.nn.Conv2d(output_ch, feature, 3, 1, 1)
        self.feature_map = torch.nn.ModuleDict({f'Conv_{i}': torch.nn.Conv2d(feature, feature, 3, 1, 1) for i in range(block_num - 1)})
        self.activations = torch.nn.ModuleDict({f'Conv_{i}': activations[activation]() for i in range(block_num - 1)})
        self.residual_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.start_conv(x)
        # x_in = self.residual_shortcut(x)
        x_in = x
        x = self.start_activation(self.patch_extraction(x))
        for (layer), (activation) in zip(self.feature_map.values(), self.activations.values()):
            x = activation(layer(x))
        if self.residual:
            output = self.residual_conv(x) + x_in
        else:
            output = self.residual_conv(x)
        return output

    def get_feature(self, x: torch.Tensor, pick_layer: list) -> dict:

        return_features = {}
        x = self.start_conv(x)
        x_in = x
        if 'start_conv' in pick_layer:
            return_features['start_conv'] = x_in
        for (layer_name, layer), (activation_name, activation) in zip(self.feature_map.items(), self.activations.items()):
            x = activation(layer(x))
            if activation_name in pick_layer:
                return_features[activation_name] = x
        if self.residual:
            output = self.residual_conv(x) + x_in
        else:
            output = self.residual_conv(x)
        if 'output_conv' in pick_layer:
            return_features['output_conv'] = output
        return return_features

    def plot_feature(self, x: torch.Tensor, *args,
                     save_dir: str='HSCNN_features', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        color_mode = kwargs.get('color_mode', False)
        save_color_dir = kwargs.get('save_color_dir', 'HSCNN_color')
        data_name = kwargs.get('data_name', 'CAVE')
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        os.makedirs(save_dir, exist_ok=True)
        if color_mode is True:
            os.makedirs(save_color_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            # nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            feature = normalize(feature.permute((1, 0, 2, 3)))  # .clamp(0., 1.)
            torchvision.utils.save_image(feature, os.path.join(save_dir, f'{layer_name}.png'),
                                         nrow=row, padding=0)
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature})
            if color_mode is True:
                self.plot_color_img(nd_feature, os.path.join(save_color_dir, f'{layer_name}.png'),
                                    mode=data_name)

    def plot_diff(self, x: torch.Tensor, label: torch.Tensor, *args,
                  save_dir: str='HSCNN_diff', **kwargs) -> None:

        mat_mode = kwargs.get('mat_mode', False)
        pick_layers = kwargs.get('pick_layers', ['start_conv'] + list(self.activations.keys()) + ['output_conv'])
        _, ch, h, w = label.shape
        row, col = int(np.ceil(np.sqrt(self.output_ch))), int(np.ceil(np.sqrt(self.output_ch)))
        plot_array = np.zeros((h * row, col * w))
        os.makedirs(save_dir, exist_ok=True)
        features = self.get_feature(x, pick_layers)
        for layer_name, feature in features.items():
            plot_array[:] = 0.
            feature = (feature.clamp(0., 1.) - label).abs()
            feature_mean = feature.mean(dim=(-1, -2))
            nd_feature = feature.squeeze().detach().numpy().transpose(1, 2, 0)
            for i in range(row):
                for j in range(col):
                    if i * row + j >= ch: break
                    plot_array[h * i: h * (i + 1), w * j: w * (j + 1)] = nd_feature[:, :, i * row + j]
            plt.imsave(os.path.join(save_dir, f'{layer_name}.png'), plot_array, cmap='jet')
            feature_mean = feature_mean.squeeze().detach().numpy()
            if mat_mode is True:
                scipy.io.savemat(os.path.join(save_dir, f'{layer_name}.mat'), {'data': nd_feature, 'mean': feature_mean})
        plot_label = label.permute((1, 0, 2, 3))
        torchvision.utils.save_image(plot_label, os.path.join(save_dir, 'output_img.png'),
                                        nrow=row, padding=0)

    def plot_color_img(self, input_img: np.ndarray,
                       save_name: str, *args, mode='CAVE', **kwargs):
        func_name = transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
        if mode == 'CAVE' or mode == 'ICVL':
            start_wave = 400
            last_wave = 700
        else:
            start_wave = 420
            last_wave = 720
        x = np.arange(start_wave, last_wave + 1, 10)
        trans_filter = func_name(x)
        ch = x.shape[0]
        row = int(np.ceil(np.sqrt(ch)))
        all_img = []
        for i, ch in enumerate(range(start_wave, last_wave + 1, 10)):
            trans_data = np.expand_dims(input_img[:, :, i], axis=-1).dot(np.expand_dims(trans_filter[i], axis=0)).clip(0., 1.)
            all_img.append(trans_data)
        tensor_img = torch.Tensor(np.array(all_img)).permute(0, 3, 1, 2)
        torchvision.utils.save_image(tensor_img, save_name, nrow=row, padding=0)

if __name__ == '__main__':

    model = HSCNN(1, 31, activation='sin2')
    summary(model, (1, 64, 64))
