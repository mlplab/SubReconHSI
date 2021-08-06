# coding: utf-8


import os
import sys
import argparse
import datetime
import torch
import torchvision
from torchinfo import summary
from trainer import HiNASTrainer, Trainer
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.HiNAS import TrainHiNAS
from model.HiNAS import SearchHiNAS as PreHiNAS
from model.layers import MSE_SAMLoss
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=100, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HiNAS', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=3, type=int, help='Model Block Number')
parser.add_argument('--node_num', '-nn', default=3, type=int, help='Model Node Number')
parser.add_argument('--cell_num', '-cn', default=4, type=int, help='Model Cell Number')
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start training time')
parser.add_argument('--loss', '-l', default='mse', type=str, help='Loss Mode')
parser.add_argument('--layer_id', '-li', nargs='*', default=['normal', 'depth', 'dilate', 'deform', 'residual', 'mix', 'ghost'],
# parser.add_argument('--layer_id', '-li', nargs='*', default=['normal'],
                    type=list, help='Search layer id')
args = parser.parse_args()


dt_now = args.start_time
batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num
ratio = 2
mode = 'normal'
loss_mode = args.loss
default_layerid = args.layer_id


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


img_path = f'../SCI_dataset/My_{data_name}_128'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('../SCI_result', f'{data_name}_{dt_now}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
filter_path = os.path.join('../SCI_dataset', 'D700_CSF.mat')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now}')
all_trained_ckpt_path = os.path.join(ckpt_path, 'all_trained_sota')
os.makedirs(all_trained_ckpt_path, exist_ok=True)


model_obj = {'HSCNN': HSCNN, 'HyperReconNet': HyperReconNet, 'DeepSSPrior': DeepSSPrior}
activations = {'HSCNN': 'leaky', 'HyperReconNet': 'relu', 'DeepSSPrior': 'relu'}


save_model_name = f'{model_name}_{block_num:02d}_{loss_mode}_{dt_now}_{concat_flag}'
if os.path.exists(os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar')):
    print(f'already trained {save_model_name}')
    sys.exit(0)


train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
train_dataset = PatchMaskDataset(train_path, mask_path, transform=train_transform, concat=concat_flag)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


test_transform = None
test_dataset = PatchMaskDataset(test_path, mask_path, transform=test_transform, concat=concat_flag)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = model_obj[model_name](input_ch, 31, feature_num=31, block_num=block_num,
                              layer_num=block_num, activation=activations[model_name])


model.to(device)
summary(model, (1, input_ch, 48, 48), depth=8)
criterions = {'mse': torch.nn.MSELoss, 'mse_sam': MSE_SAMLoss}
criterion = criterions[loss_mode]().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, .5)


ckpt_cb = ModelCheckPoint(ckpt_path, save_model_name,
                          mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, 
                  callbacks=[ckpt_cb], device=device)
train_loss, val_loss = trainer.train(epochs, train_dataloader, test_dataloader)
torch.save({'model_state_dict': model.state_dict(),
            'optim': optim.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'epoch': epochs},
           os.path.join(all_trained_ckpt_path, f'{save_model_name}.tar'))
