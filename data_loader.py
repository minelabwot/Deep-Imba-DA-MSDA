import torch.utils.data as data
import torch
from PIL import Image
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.utils.data as Data
import numpy as np

BATCH_SIZE = 128


def npy_loader(path):
    return torch.from_numpy(np.load(path))


def load_data(src, tar, batch_size=128):
    print('data load,source:{},target:{}'.format(src, tar))
    fault_dir_src_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_feature.npy'.format(
        src)
    fault_dir_src_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_label.npy'.format(
        src)
    fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_feature.npy'.format(
        tar)
    fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_{}_label.npy'.format(
        tar)
    dataset_source = Data.TensorDataset(npy_loader(fault_dir_src_feature), npy_loader(fault_dir_src_label))
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    dataset_target = Data.TensorDataset(npy_loader(fault_dir_tar_feature), npy_loader(fault_dir_tar_label))
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataloader_source, dataloader_target


def load_test_data(tar, batch_size=128):
    # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_feature.npy'
    # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_label.npy'
    # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_feature.npy'
    # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_label.npy'
    fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/test_dataset/motor_{}_feature.npy'.format(
        tar)
    fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/test_dataset/motor_{}_label.npy'.format(
        tar)
    torch_feature_tar = npy_loader(fault_dir_tar_feature)
    torch_label_tar = npy_loader(fault_dir_tar_label)
    dataset = TensorDataset(torch_feature_tar, torch_label_tar)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,

    )
    return dataloader


def load_validate_data(tar,  batch_size=128):
    fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/validate_dataset/motor_{}_feature.npy'.format(
        tar)
    fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/validate_dataset/motor_{}_label.npy'.format(
        tar)
    torch_feature_tar = npy_loader(fault_dir_tar_feature)
    torch_label_tar = npy_loader(fault_dir_tar_label)
    dataset = TensorDataset(torch_feature_tar, torch_label_tar)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,

    )
    return dataloader

# def npy_loader(path):
#     return torch.from_numpy(np.load(path))
# 
# 
# def load_data(batch_size=128):
#     fault_dir_src_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_0_feature.npy'
#     fault_dir_src_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_0_label.npy'
#     # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_feature.npy'
#     # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_label.npy'
#     # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_feature.npy'
#     # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_label.npy'
#     fault_dir_tar_feature = '/home/zhk/jupyterproject/motor-2-feature.npy'
#     fault_dir_tar_label = '/home/yshuyan/project/bearing/dataset_th/motor_2_label.npy'
#     dataset_source = Data.TensorDataset(npy_loader(fault_dir_src_feature), npy_loader(fault_dir_src_label))
#     dataloader_source = torch.utils.data.DataLoader(
#         dataset=dataset_source,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#     )
#     dataset_target = Data.TensorDataset(npy_loader(fault_dir_tar_feature), npy_loader(fault_dir_tar_label))
#     dataloader_target = torch.utils.data.DataLoader(
#         dataset=dataset_target,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#     )
#     return dataloader_source, dataloader_target
# 
# 
# def load_test_data(batch_size=128):
#     # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_feature.npy'
#     # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_1_label.npy'
#     # fault_dir_tar_feature = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_feature.npy'
#     # fault_dir_tar_label = '/home/zhk/transferLearning/bearing_cppy/bearing/DANN-inner/dataset/motor_3_label.npy'
#     fault_dir_tar_feature = '/home/zhk/jupyterproject/motor-2-feature.npy'
#     fault_dir_tar_label = '/home/yshuyan/project/bearing/dataset_th/motor_2_label.npy'
#     torch_feature_tar = npy_loader(fault_dir_tar_feature)
#     torch_label_tar = npy_loader(fault_dir_tar_label)
#     dataset = TensorDataset(torch_feature_tar, torch_label_tar)
#     dataloader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
# 
#     )
#     return dataloader
