# coding: utf-8


import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from evaluate import PSNRMetrics, SAMMetrics, None_Evaluate
from pytorch_ssim import SSIM
from utils import normalize


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if device == 'cuda':
#     torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=8)


class Trainer(object):

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, *args, scheduler=None,
                 callbacks=None, device: str='cpu', **kwargs):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.device = device
        self.use_amp = kwargs.get('use_amp', False)
        self.psnr = kwargs.get('psnr', None_Evaluate())  # SNRMetrics().eval())
        self.sam = kwargs.get('sam', None_Evaluate())  # SAMMetrics().eval())
        self.ssim = kwargs.get('ssim', None_Evaluate())  # SSIM().eval()
        self.colab_mode = kwargs.get('colab_mode', False)
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, epochs: int, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader, init_epoch: int=0) -> (np.ndarray, np.ndarray):

        if self.colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        train_output = []
        val_output = []

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            print(dt_now)
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            show_mean = self._all_step(val_dataloader, mode=mode, desc_str=desc_str, columns=columns)
            train_output.append(show_mean)

            self.model.eval()
            mode = 'Val'
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            show_mean = self._all_step(val_dataloader, mode=mode, desc_str=desc_str, columns=columns)
            val_output.append(show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True,
                                      device=self.device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('=' * int(columns))

        train_output = np.array(train_output)
        val_output = np.array(val_output)
        return train_output, val_output

    def _trans_data(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.device)

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor,
              train: bool=True) -> (torch.Tensor, torch.Tensor):
        with torch.cuda.amp.autocast(self.use_amp):
            output = self.model(inputs)
            loss = self.criterion(output, labels)
        if train is True:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        return loss, output

    def _step_show(self, pbar, *args, **kwargs) -> None:
        if self.device == 'cuda':
            kwargs['Allocate'] = f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache'] = f'{torch.cuda.memory_reserved(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self

    def _evaluate(self, output: torch.Tensor, label: torch.Tensor) -> (float, float, float):
        output = output.float().to(self.device)
        output = torch.clamp(output, 0., 1.)
        labels = torch.clamp(label, 0., 1.)
        return [self.psnr(labels, output).item(),
                self.ssim(labels, output).item(),
                self.sam(labels, output).item()]

    def _all_step(self, dataloader, mode: str, desc_str: str, columns: int) -> np.ndarray:
        step_loss = []
        step_eval = []
        with tqdm(dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs = self._trans_data(inputs)
                labels = self._trans_data(labels)
                if mode.lower() == 'train':
                    loss, output = self._step(inputs, labels)
                elif mode.lower() == 'val':
                    with torch.no_grad():
                        loss, output = self._step(inputs, labels, train=False)
                step_loss.append(loss.item())
                show_loss = np.mean(val_loss)
                step_eval.append(self._evaluate(output, labels))
                show_mean = np.mean(show_val_eval, axis=0)
                evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                torch.cuda.empty_cache()
        show_loss, show_mean = np.mean(step_loss), np.mean(step_eval, axis=0)
        show_mean = np.insert(show_mean, 0, show_loss)
        return show_mean
