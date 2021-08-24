# coding: utf-8


import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
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

class TPUTrainer(Trainer):

    def multi_train(self, epochs: int, batch_size: int, train_dataloader,
                    eval_dataloader, *args,
                    init_epoch: int=0, **kwargs) -> (np.ndarray, np.ndarray):

        flush_time = kwargs.get('flush_time', 10)
        train_dataloader_num = len(train_dataloader)
        val_dataloader_num = len(eval_dataloader)
        if flush_time > train_dataloader_num or flush_time > val_dataloader_num:
            train_flush_time = 2
            val_flush_time = 2
        else:
            train_flush_time = train_dataloader_num // flush_time
            val_flush_time = val_dataloader_num // flush_time

        if self.colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        train_output = []
        val_output = []

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            xm.master_print(dt_now)
            train_loss = []
            val_loss = []
            ####################################################################
            # Train
            mode = 'Train'
            self.model.train()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            para_dataloader = pl.ParallelLoader(train_dataloader, [self.device]).per_device_loader(self.device)
            show_mean = self._multi_all_step(para_dataloader, train_flush_time, mode=mode, desc_str=desc_str)
            train_output.append(show_mean)
            xm.master_print('-' * int(columns))
            ####################################################################
            # Val
            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            para_dataloader = pl.ParallelLoader(eval_dataloader, [self.device]).per_device_loader(self.device)
            show_mean = self._multi_all_step(para_dataloader, val_flush_time, mode=mode, desc_str=desc_str)
            val_output.append(show_mean)
            ####################################################################
            xm.master_print('=' * int(columns))
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True,
                                      device=self.device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()

        train_output = np.array(train_output)
        val_output = np.array(val_output)
        return train_output, val_output

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor,
              train: bool=True) -> (torch.Tensor, torch.Tensor):
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            xm.optimizer_step(self.optimizer, barrier=True)
            self.optimizer.zero_grad()
        return loss, output

    def _multi_all_step(self, dataloader, flush_time: int, mode: str, desc_str: str) -> (list, list):
        dataloader_num = len(dataloader)
        step_loss = []
        step_eval = []
        start_time = time.time()
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = self._trans_data(inputs)
            labels = self._trans_data(labels)
            if mode.lower() == 'train':
                loss, output = self._step(inputs, labels)
            elif mode.lower() == 'val':
                with torch.no_grad():
                    loss, output = self._step(inputs, labels, train=False)
            step_loss.append(loss.item())
            show_loss = np.mean(step_loss)
            step_eval.append(self._evaluate(output, labels))
            show_mean = np.mean(step_eval, axis=0)
            evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
            if i % flush_time == 0:
                now_time = time.time() - start_time
                now_h, now_m, now_s = int(now_time // 3600), int(now_time // 60), now_time % 60
                bar = '#' * (i // flush_time) + '.' * ((dataloader_num - i) // flush_time)
                progress = '| '.join([desc_str, f'Time: {now_h:02d}:{now_m:02d}:{now_s:06.3f}',
                                        f'{i:05d} / {dataloader_num:05d}',
                                        bar,
                                        f'Loss: {show_loss:.7f}',
                                        f'Evaluate: {evaluate}'])
                xm.master_print(progress)
        show_loss, show_mean = np.mean(step_loss), np.mean(step_eval, axis=0)
        show_mean = np.insert(show_mean, 0, show_loss)
        return show_mean
