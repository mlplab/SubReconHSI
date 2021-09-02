# coding: UTF-8


import os
import numpy as np
import scipy.io as sio
import torch
import torchvision


class PatchMaskDataset(torch.utils.data.Dataset):

    def __init__(self, img_path: str, mask_path: str, *args, concat: bool=False,
                 data_key: str='data', transform=None, **kwargs) -> None:

        self.img_path = img_path
        self.data = os.listdir(img_path)
        self.mask_path = mask_path
        self.data_len = len(self.data)
        self.concat = concat
        self.data_key = data_key
        self.transforms = transform
        self.mask_transforms = torchvision.transforms.ToTensor()

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data

    def __len__(self):
        return self.data_len


class PatchEvalDataset(PatchMaskDataset):

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)
        if self.concat is True:
            input_data = torch.cat([measurement_data, mask], dim=0)
        else:
            input_data = measurement_data
        return self.data[idx], input_data, label_data


class SpectralFusionDataset(torch.utils.data.Dataset):

    def __init__(self, img_path: str, mask_path: str, *args, data_name='CAVE',
                 concat: bool=False, data_key: str='data', transform=None,
                 # return_mode: str='dict',
                 rgb_input: bool=True, rgb_label: bool=True, **kwargs) -> None:

        self.img_path = img_path
        self.data = os.listdir(img_path)
        self.mask_path = mask_path
        self.data_len = len(self.data)
        self.concat = concat
        self.data_key = data_key
        self.transforms = transform
        self.mask_transforms = torchvision.transforms.ToTensor()
        self.data_name = data_name
        # self.return_mode = return_mode
        self.rgb_input = rgb_input
        self.rgb_label = rgb_label
        self.rgb_ch = {'CAVE': (26, 16, 9),
                       'Harvard': (26, 16, 9),
                       'ICVL': (26, 16, 9)}

    def __getitem__(self, idx: int) -> (dict, dict):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)

        if self.concat is True:
            hsi_data = torch.cat([measurement_data, mask], dim=0)
        else:
            hsi_data = measurement_data
        hsi_label = trans_data

        if self.rgb_input:
            rgb_input = trans_data[self.rgb_ch[self.data_name], :, :]
        else:
            rgb_input = torch.empty_like(measurement_data)
        if self.rgb_label:
            rgb_label = trans_data[self.rgb_ch[self.data_name], :, :]
        else:
            rgb_label = torch.empty_like(measurement_data)

        input_data = {'hsi': hsi_data}
        label_data = {'hsi': hsi_label}
        input_data['rgb'] = rgb_input
        label_data['rgb'] = rgb_label

        return input_data, label_data

    def __len__(self):
        return self.data_len


class SpectralFusionEvalDataset(SpectralFusionDataset):


    def __getitem__(self, idx: int) -> (dict, dict):
        patch_id = self.data[idx].split('.')[0].split('_')[-1]
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))[self.data_key]
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        label_data = trans_data
        mask = sio.loadmat(os.path.join(self.mask_path, f'mask_{patch_id}.mat'))[self.data_key]
        mask = self.mask_transforms(mask)
        measurement_data = (trans_data * mask).sum(dim=0, keepdim=True)

        if self.concat is True:
            hsi_data = torch.cat([measurement_data, mask], dim=0)
        else:
            hsi_data = measurement_data
        hsi_label = trans_data

        if self.rgb_input:
            rgb_input = trans_data[self.rgb_ch[self.data_name], :, :]
        else:
            rgb_input = torch.empty_like(measurement_data)
        if self.rgb_label:
            rgb_label = trans_data[self.rgb_ch[self.data_name], :, :]
        else:
            rgb_label = torch.empty_like(measurement_data)

        input_data = {'hsi': hsi_data}
        label_data = {'hsi': hsi_label}
        input_data['rgb'] = rgb_input
        label_data['rgb'] = rgb_label

        return self.data[idx], input_data, label_data

    def __len__(self):
        return self.data_len
