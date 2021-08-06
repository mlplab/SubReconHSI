# coding: utf-8


import os
import shutil
import scipy.io
import warnings
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import mpl_toolkits
import mpl_toolkits.axes_grid1
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import pytorch_ssim
from utils import normalize


warnings.simplefilter('ignore')
device = 'cpu'
plt.rcParams['image.cmap'] = 'gray'


def compare_sam(x: np.ndarray, y: np.ndarray) -> float:
    x_sqrt = np.linalg.norm(x, axis=-1)
    y_sqrt = np.linalg.norm(y, axis=-1)
    xy = (x * y).sum(axis=-1)
    metrics = xy / (x_sqrt * y_sqrt + 1e-6)
    angle = np.arccos(metrics)
    return angle.mean()


class RMSEMetrics(torch.nn.Module):

    def __init__(self) -> None:
        super(RMSEMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss().eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.criterion(x, y))


class PSNRMetrics(torch.nn.Module):

    def __init__(self) -> None:
        super(PSNRMetrics, self).__init__()
        self.criterion = torch.nn.MSELoss().eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 10. * torch.log10(1. / self.criterion(x, y))


class SAMMetrics(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        xy = torch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class Evaluater(object):

    def __init__(self, data_name: str, save_img_path: str='output_img', 
                 save_mat_path: str='output_mat', save_csv_path: str='output_csv', 
                 **kwargs) -> None:
        self.data_name = data_name
        self.save_alls_path = save_img_path
        self.save_mat_path = save_mat_path
        self.save_csv_path = save_csv_path
        self.output_ch = {'CAVE': (26, 16, 9), 'Harvard': (26, 16, 9), 'ICVL': (26, 16, 9)}
        os.makedirs(self.save_alls_path, exist_ok=True)
        os.makedirs(save_mat_path, exist_ok=True)

    def _plot_img(self, ax: plt.Axes, img: np.ndarray, title: str='None', 
                  colorbar: bool=False) -> None:
        if colorbar is not False:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')
            im = ax.imshow(img, cmap='jet')
            plt.colorbar(im, cax=cax)
        else:
            im = ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        return self

    def _save_all(self, i: int, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        save_alls_path = 'save_all'
        _, c, h, w = outputs.size()
        diff = torch.abs(outputs - labels).squeeze().numpy()
        diff = diff.transpose(1, 2, 0).mean(axis=-1)
        diff = normalize(diff)
        inputs = normalize(inputs.squeeze().numpy())
        outputs = outputs.squeeze().numpy().transpose(1, 2, 0)
        outputs = normalize(outputs[:, :, self.output_ch[self.data_name]])
        labels = labels.squeeze().numpy().transpose(1, 2, 0)
        labels = normalize(labels[:, :, self.output_ch[self.data_name]])
        fig_num = 4
        plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 4, 1)
        if inputs.shape[0] == 32:
            inputs = inputs[0]
        ax.imshow(inputs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('input')
        figs = [outputs, labels]
        titles = ['output', 'label']
        for j, (fig, title) in enumerate(zip(figs, titles)):
            ax = plt.subplot(1, fig_num, j + 2)
            self._plot_img(ax, fig, title)
        ax = plt.subplot(1, fig_num, fig_num)
        self._plot_img(ax, diff, title='diff', colorbar=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_alls_path, f'output_alls_{i}.png'), bbox_inches='tight')
        plt.close()
        return self

    def _save_mat(self, i: int, idx: int, output: torch.Tensor) -> None:
        output_mat = output.squeeze().to('cpu').detach().numpy().copy()
        output_mat = output_mat.transpose(1, 2, 0)
        scipy.io.savemat(os.path.join(self.save_mat_path, f'{i:05d}.mat'), {'data': output_mat, 'idx': idx})
        return self

    def _save_csv(self, output_evaluate: list, header: list) -> None:
        header.append('Time')
        output_evaluate_np = np.array(output_evaluate, dtype=np.float32)
        means = list(np.mean(output_evaluate_np, axis=0))
        output_evaluate.append(means)
        output_evaluate_csv = pd.DataFrame(output_evaluate)
        output_evaluate_csv.to_csv(self.save_csv_path, header=header)
        print(means)
        return self

    def _step_show(self, pbar, *args, **kwargs) -> None:
        if device == 'cuda':
            kwargs['Allocate'] = f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache'] = f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self


class ReconstEvaluater(Evaluater):

    def metrics(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                evaluate_fn: list[torch.nn.Module, torch.nn.Module, torch.nn.Module], 
                header: object=None, colab_mode: bool=False) -> None:
        model.eval()
        output_evaluate = []
        if colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        with torch.no_grad():
            with tqdm(dataset, ncols=columns, ascii=True) as pbar:
                for i, (idx, inputs, labels) in enumerate(pbar):
                    evaluate_list = []
                    inputs = inputs.unsqueeze(0).to(device)
                    labels = labels.unsqueeze(0).to(device)
                    output = model(inputs)
                    metrics_output = torch.clamp(output, min=0., max=1.)
                    metrics_labels = torch.clamp(labels, min=0., max=1.)
                    for metrics_func in evaluate_fn:
                        metrics = metrics_func(metrics_output, metrics_labels)
                        evaluate_list.append(f'{metrics.item():.7f}')
                    evaluate_list.append(f'{finish_time:.5f}')
                    output_evaluate.append(evaluate_list)
                    show_evaluate = np.mean(np.array(output_evaluate, dtype=np.float32), axis=0)
                    self._step_show(pbar, Metrics=show_evaluate)
                    del show_evaluate
                    self._save_all(i, inputs, output, labels)
                    self._save_mat(i, idx, output)
        self._save_csv(output_evaluate, header)
        return self
