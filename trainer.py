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
            show_train_eval = []
            show_val_eval = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(train_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs = self._trans_data(inputs)
                    labels = self._trans_data(labels)
                    loss, output = self._step(inputs, labels)
                    train_loss.append(loss.item())
                    show_loss = np.mean(train_loss)
                    show_train_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_train_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            train_output.append(show_mean)

            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(val_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs = self._trans_data(inputs)
                    labels = self._trans_data(labels)
                    with torch.no_grad():
                        loss, output = self._step(inputs, labels, train=False)
                    val_loss.append(loss.item())
                    show_loss = np.mean(val_loss)
                    show_val_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_val_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            val_output.append(show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True,
                                      device=self.device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('-' * int(columns))

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

    def _evaluate(self, output, label) -> (float, float, float):
        output = output.float().to(self.device)
        output = torch.clamp(output, 0., 1.)
        labels = torch.clamp(label, 0., 1.)
        return [self.psnr(labels, output).item(),
                self.ssim(labels, output).item(),
                self.sam(labels, output).item()]


class TPUTrainer(Trainer):

    def single_train(self, epochs: int,
                     train_dataloader: torch.utils.data.DataLoader,
                     val_dataloader: torch.utils.data.DataLoader,
                     init_epoch: int=0) -> (np.ndarray, np.ndarray):

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
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            show_train_eval = []
            show_val_eval = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(train_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs = self._trans_data(inputs)
                    labels = self._trans_data(labels)
                    loss, output = self._step(inputs, labels)
                    train_loss.append(loss.item())
                    show_loss = np.mean(train_loss)
                    show_train_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_train_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            train_output.append(show_mean)

            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(val_dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs = self._trans_data(inputs)
                    labels = self._trans_data(labels)
                    with torch.no_grad():
                        loss, output = self._singel_step(inputs, labels, train=False)
                    val_loss.append(loss.item())
                    show_loss = np.mean(val_loss)
                    show_val_eval.append(self._evaluate(output, labels))
                    show_mean = np.mean(show_val_eval, axis=0)
                    evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                    self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                    torch.cuda.empty_cache()
            show_mean = np.insert(show_mean, 0, show_loss)
            val_output.append(show_mean)
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

    def multi_train(self, epochs: int, batch_size: int, train_dataloader,
                    eval_dataloader, *args, nprocs: int=8,
                    init_epoch: int=0, **kwargs) -> None:

        train_dataloader_num = len(train_dataloader)
        val_dataloader_num = len(eval_dataloader)
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
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            show_train_eval = []
            show_val_eval = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            train_start_time = time.time()
            para_dataloader = pl.ParallelLoader(train_dataloader, [self.device]).per_device_loader(self.device)
            flush_time = 0
            for i, (inputs, labels) in enumerate(para_dataloader):
                inputs = self._trans_data(inputs)
                labels = self._trans_data(labels)
                loss, output = self._step(inputs, labels)
                train_loss.append(loss.item())
                show_loss = np.mean(train_loss)
                show_train_eval.append(self._evaluate(output, labels))
                show_mean = np.mean(show_train_eval, axis=0)
                evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                # self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                if i % (batch_size // nprocs) == 0:
                    now_time = time.time() - train_start_time
                    now_h, now_m, now_s = int(now_time // 3600), int(now_time // 60), now_time % 60
                    bar = '#' * i + '.' * (train_dataloader_num - i)
                    progress = '| '.join([desc_str, f'Time: {now_h:02d}:{now_m:02d}:{now_s:06.3f}',
                                          bar,
                                          f'{i:05d} / {train_dataloader_num:05d}',
                                          f'Loss: {show_loss:.7f}'])  # | Evaluate: {evaluate}'])
                    xm.master_print(progress)
            show_mean = np.insert(show_mean, 0, show_loss)
            train_output.append(show_mean)

            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            val_start_time = time.time()
            val_dataloader = pl.ParallelLoader(eval_dataloader, [self.device]).per_device_loader(self.device)
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = self._trans_data(inputs)
                labels = self._trans_data(labels)
                with torch.no_grad():
                    loss, output = self._step(inputs, labels, train=False)
                val_loss.append(loss.item())
                show_loss = np.mean(val_loss)
                show_val_eval.append(self._evaluate(output, labels))
                show_mean = np.mean(show_val_eval, axis=0)
                evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                # self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                if i % (batch_size // nprocs) == 0:

                    now_time = time.time() - val_start_time
                    now_h, now_m, now_s = int(now_time // 3600), int(now_time // 60), now_time % 60
                    bar = '#' * i + '.' * (val_dataloader_num - i)
                    progress = '| '.join([desc_str, f'Time: {now_h:02d}:{now_m:02d}:{now_s:06.3f}',
                                            bar,
                                            f'{i:05d} / {val_dataloader_num:05d}',
                                            f'Loss: {show_loss:.7f}'])  # | Evaluate: {evaluate}'])
                    xm.master_print(progress)
            show_mean = np.insert(show_mean, 0, show_loss)
            val_output.append(show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True,
                                      device=self.device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            xm.master_print('-' * int(columns))

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

