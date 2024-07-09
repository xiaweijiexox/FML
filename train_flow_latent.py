# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import shutil
from functools import partial
from time import time
import math 
import pickle

import torch
from omegaconf import OmegaConf
import numpy as np
from ddp_utils import init_processes
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets_prep import get_dataset
from EMA import EMA
from models import create_network
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.tensorboard import SummaryWriter
from math import pi
import numpy as np
from scipy.optimize import root_scalar

import numpy as np
from scipy.optimize import root_scalar
import torch.nn as nn
import torchvision.models as models
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from diffusers.models import AutoencoderKL
from collections import OrderedDict
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import gc
import psutil
def compute_distance(tensor1, tensor2):
    # 计算两个tensor之间的距离，这里使用Euclidean距离
    return torch.norm(tensor1 - tensor2).item()
import random
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor
import linecache
import ast

def compute_distance(tensor1, tensor2):
    # 计算两个tensor之间的距离，这里使用Euclidean距离
    return torch.norm(tensor1 - tensor2).item()

def sliced_wasserstein_distance(sample1, sample2, num_projections=200):
    """
    计算两个样本之间的切片Wasserstein距离。

    参数:
    sample1 (torch.Tensor): 第一个样本张量 (N, D)
    sample2 (torch.Tensor): 第二个样本张量 (M, D)
    num_projections (int): 投影的数量 (默认为100)

    返回:
    float: 切片Wasserstein距离
    """
    # 获取样本的特征维度
    d = sample1.size(1)

    # 在球面上均匀采样投影方向
    projections = torch.randn(num_projections, d).to(sample1.device)
    projections = F.normalize(projections, dim=1)

    # 对每个投影计算投影样本点
    proj_sample1 = sample1 @ projections.T
    proj_sample2 = sample2 @ projections.T

    # 计算每个投影上的Wasserstein距离
    wasserstein_distances = []
    for i in range(num_projections):
        proj_dist = torch.sort(proj_sample1[:, i])[0] - torch.sort(proj_sample2[:, i])[0]
        wasserstein_distances.append(proj_dist.abs().mean())

    # 返回所有投影上的平均Wasserstein距离
    return torch.stack(wasserstein_distances).mean().item()

def compute_distance_matrix(list1, list2, start_idx, end_idx):
    block_size_actual = end_idx - start_idx
    distance_matrix = np.zeros((block_size_actual, block_size_actual))
    
    for i in range(block_size_actual):
        for j in range(block_size_actual):
            distance_matrix[i, j] = sliced_wasserstein_distance(list1[start_idx + i], list2[start_idx + j])
    
    return distance_matrix

def reorder_list(list1, list2, block_size):
    # 先随机打乱 list1
    list_index = [i for i in range(len(list1))]
    list_index_ran = list_index.copy()
    random.shuffle(list_index_ran)
    dict_for = dict(zip(list_index,list_index_ran))
    
    
    n = len(list1)
    num_blocks = (n + block_size - 1) // block_size  # 计算分块数量
    
    reordered_list2 = []
    
    for block_idx in range(num_blocks):
        # 计算每个分块的起始和结束索引
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n)
        
        block_size_actual = end_idx - start_idx
        distance_matrix = np.zeros((block_size_actual, block_size_actual))
        
        for i in range(block_size_actual):
            for j in range(block_size_actual):
                distance_matrix[i, j] = sliced_wasserstein_distance(list1[dict_for[start_idx + i]], list2[dict_for[start_idx + j]])
        
        # 使用线性分配算法找到最小化总距离的排列
        row_ind, col_ind = linear_sum_assignment(distance_matrix)        
        # 重新排序当前分块的list2
        reordered_block = [list2[start_idx + j] for j in col_ind]
        reordered_list2.extend(reordered_block)  #最终会得到随机距离矩阵
    
    # 现在还原随机的顺序
    reordered_list_2_copy = reordered_list2.copy()
    for m in range(len(reordered_list2)):
        reordered_list_2_copy[dict_for[m]] = reordered_list2[m]
    
    return reordered_list_2_copy
    

def reorder_list_conv(list1, list2, block_size):
    n = len(list1)
    
    # 先随机打乱 list1
    list_index = [i for i in range(len(list1))]
    list_index_ran = list_index.copy()
    random.shuffle(list_index_ran)
    dict_for = dict(zip(list_index,list_index_ran))
    
    num_blocks =  (n-3 + (block_size -3) -1) // (block_size-3) 
    
    #选取上一轮已经抽过的两个索引
    for block_idx in range(num_blocks):
        # 计算每个分块的起始和结束索引
        start_idx = block_idx * (block_size-3)
        end_idx = min(start_idx + block_size, n)
        sample_indices = [dict_for[i] for i in range(start_idx,end_idx)]
        
        
        block_size_actual = end_idx - start_idx
        distance_matrix = np.zeros((block_size_actual, block_size_actual))
        
        for i in range(block_size_actual):
            for j in range(block_size_actual):
                distance_matrix[i, j] = compute_distance(list1[sample_indices[i]], list2[sample_indices[j]])
        
        # 使用线性分配算法找到最小化总距离的排列 12345---54132
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        
        # 重新排序当前抽取样本的list2 ——54132
        reordered_sample = [list2[sample_indices[j]] for j in col_ind]
        
        
        # 将重新排序的样本放回原来的位置
        reordered_list2 = list2.copy()
        for idx, new_idx in zip(sample_indices, col_ind):
            reordered_list2[idx] = list2[sample_indices[new_idx]]
        
        list2 = reordered_list2

    return reordered_list2



    
def reorder_list_conv_forback(list1, list2, block_size):
    n = len(list1)
    
    # 先随机打乱 list1
    list_index = [i for i in range(len(list1))]
    list_index_ran = list_index.copy()
    random.shuffle(list_index_ran)
    dict_for = dict(zip(list_index,list_index_ran))
    
    num_blocks =  (n-2 + (block_size -2) -1) // (block_size-2) 
    
    #选取上一轮已经抽过的两个索引
    for block_idx in range(num_blocks):
        # 计算每个分块的起始和结束索引
        start_idx = block_idx * (block_size-2)
        end_idx = min(start_idx + block_size, n)
        sample_indices = [dict_for[i] for i in range(start_idx,end_idx)]
        
        
        block_size_actual = end_idx - start_idx
        distance_matrix = np.zeros((block_size_actual, block_size_actual))
        
        for i in range(block_size_actual):
            for j in range(block_size_actual):
                distance_matrix[i, j] = compute_distance(list1[sample_indices[i]], list2[sample_indices[j]])
        
        # 使用线性分配算法找到最小化总距离的排列 12345---54132
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        
        # 重新排序当前抽取样本的list2 ——54132
        reordered_sample = [list2[sample_indices[j]] for j in col_ind]
        
        
        # 将重新排序的样本放回原来的位置
        reordered_list2 = list2.copy()
        for idx, new_idx in zip(sample_indices, col_ind):
            reordered_list2[idx] = list2[sample_indices[new_idx]]
        
        list2 = reordered_list2
    
    #反向传播
    for block_idx in range(num_blocks):
        # 唯一的变动，sample indice反过来取
        end_idx = n - block_idx * (block_size-2)
        start_idx = n - min(block_idx * (block_size-2) + block_size, n)
        sample_indices = [dict_for[i] for i in range(start_idx,end_idx)]
        
        
        block_size_actual = end_idx - start_idx
        distance_matrix = np.zeros((block_size_actual, block_size_actual))
        
        for i in range(block_size_actual):
            for j in range(block_size_actual):
                distance_matrix[i, j] = compute_distance(list1[sample_indices[i]], list2[sample_indices[j]])
        
        # 使用线性分配算法找到最小化总距离的排列 12345---54132
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        
        # 重新排序当前抽取样本的list2 ——54132
        reordered_sample = [list2[sample_indices[j]] for j in col_ind]
        
        
        # 将重新排序的样本放回原来的位置
        reordered_list2 = list2.copy()
        for idx, new_idx in zip(sample_indices, col_ind):
            reordered_list2[idx] = list2[sample_indices[new_idx]]
        
        list2 = reordered_list2
    

    return reordered_list2

# 随机抽取
def reorder_list_sample(list1, list2, sample_size, num_samples):
    n = len(list1)
    
    for _ in range(num_samples):
        # 随机抽取10个样本
        sample_indices = random.sample(range(n), sample_size)
         
        distance_matrix = np.zeros((sample_size, sample_size))
        
        for i in range(sample_size):
            for j in range(sample_size):
                distance_matrix[i, j] = compute_distance(list1[sample_indices[i]], list2[sample_indices[j]])
        
        # 使用线性分配算法找到最小化总距离的排列 12345---54132
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        
        # 重新排序当前抽取样本的list2 ——54132
        reordered_sample = [list2[sample_indices[j]] for j in col_ind]
        
        
        # 将重新排序的样本放回原来的位置
        reordered_list2 = list2.copy()
        for idx, new_idx in zip(sample_indices, col_ind):
            reordered_list2[idx] = list2[sample_indices[new_idx]]
        
        list2 = reordered_list2

    return reordered_list2


class ScaledTanh(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return (torch.tanh(x) + 1) / 2 * (self.upper - self.lower) + self.lower
    
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def perpendicular_basis(v):
    """
    计算一个向量的标准正交基向量组
    """
    # 初始化一个单位矩阵
    basis = np.eye(len(v))
    
    # 从单位矩阵中去除一个单位向量
    basis[0] -= v / np.linalg.norm(v)
    
    # 对剩余的向量进行施密特正交化
    for i in range(1, len(v)):
        for j in range(i):
            basis[i] -= np.dot(basis[i], basis[j]) * basis[j]
        basis[i] /= np.linalg.norm(basis[i])
    
    return basis
import torch
import numpy as np

#正交扰动策略
def randomize(z_1, z_0, t, var):
    # 将 PyTorch 张量从 GPU 转移到 CPU 并转换为 NumPy 数组
    z_0_ = z_0.cpu().numpy()
    z_1_ = z_1.cpu().numpy()
    t_ = t.cpu().numpy()
    # z_0_ = z_0_.reshape(-1)
    # z_1_ = z_1_.reshape(-1)
    z_t_ = np.zeros_like(z_0_)
    for i in range(z_0_.shape[0]):
        for j in range(z_0_.shape[1]):
            for k in range(z_0_.shape[2]):
                direction_vector = z_1_[i][j][k] - z_0_[i][j][k]
                #原先是分成30个变量一个单位，现在分成2个变量一个单位——我觉得复杂的协方差矩阵会带来过多的不必要的计算！
                for l in range(int(z_0_.shape[3]/2)):
                # 计算单位向量
                    unit_vector = direction_vector[2*l:2*l+1] / np.linalg.norm(direction_vector[2*l:2*l+1])
                    # 计算垂直于连线方向向量的标准正交基向量组
                    perpendicular_basis_vectors = perpendicular_basis(unit_vector)
                    # 定义在垂直方向上的方差
                    variance = var
                    # 构造协方差矩阵
                    covariance_matrix = variance * np.dot(perpendicular_basis_vectors.T, perpendicular_basis_vectors)
                    mean = (1 - t_[i]) * z_0_[i][j][k][2*l:2*l+1] + (1e-5 + (1 - 1e-5) * t_[i]) * z_1_[i][j][k][2*l:2*l+1]
                    z_t_[i][j][k][2*l:2*l+1] = np.random.multivariate_normal(mean, covariance_matrix, 1) 
                # print(mean.shape)
                # z_t_[i][j][k] = np.random.multivariate_normal(mean, covariance_matrix, 1)    
    
    shape = z_0_.shape
    z_t = torch.tensor(z_t_.astype(np.float32))
    return z_t



# 定义 f_mode(u, s)
def f_mode(u, s):
    return 1 - u - s * (np.cos(np.pi / 2 * u) ** 2 - 1 + u)


# 定义逆函数 f_mode_inverse(t, s) 的数值求解
def f_mode_inverse(t, s):
    # 定义目标函数，使得目标函数为零时 u 为我们需要的逆
    def target_function(u, t, s):
        return f_mode(u, s) - t

    # 使用 root_scalar 来求解目标函数
    solution = root_scalar(target_function, args=(t, s), bracket=[0, 1], method='bisect')
    return solution.root


# 采样函数
def sample_t(s, num_samples=1):
    samples = []
    for _ in range(num_samples):
        u = np.random.uniform(0, 1)  # 从均匀分布采样 u
        t = f_mode(u, s)  # 计算 t
        samples.append(t)
    # 将 NumPy 数组转换为 PyTorch 张量
    t_tensor = torch.tensor(samples, dtype=torch.float32,device = "cuda")
    return t_tensor


log_dir = "/home/tsinghuaair/xwj/LFM/logs"
writer = SummaryWriter(log_dir=log_dir)


# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_device(0)

print(torch.cuda.device_count())


# Utility function to add swap_parameters_with_ema to an optimizer
def add_swap_parameters_with_ema(optimizer):
    def swap_parameters_with_ema(store_params_in_ema=True):
        if hasattr(optimizer, 'swap_parameters_with_ema'):
            optimizer.swap_parameters_with_ema(store_params_in_ema)
        else:
            raise AttributeError(f"{optimizer.__class__.__name__} object has no attribute 'swap_parameters_with_ema'")
    setattr(optimizer, 'swap_parameters_with_ema', swap_parameters_with_ema)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb


def sample_from_model(model, x_0):
    t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image

def cosine_annealing_lr(epoch, eta_max, eta_min, T_max):
    fraction = (epoch % T_max) / T_max
    cosine_factor = 0.5 * (1 + np.cos(np.pi * fraction))
    lr = eta_min + 0.5 * (eta_max - eta_min) * cosine_factor
    return lr

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    
def same_keys_different_values(dict1, dict2):
    for key in dict1:
        if key in dict2 and dict1[key] != dict2[key]:
            return True
    return False

def cosine_similarity_loss(tensor_a, tensor_b,epsilon=1e-5):
    # Flatten the tensors to convert them to vectors if they are not already
    vector_a = tensor_a.flatten()
    vector_b = tensor_b.flatten()
    
    # Calculate dot product
    dot_product = torch.dot(vector_a, vector_b)
    
    # Calculate norms (lengths) of the vectors
    norm_a = torch.norm(vector_a) + epsilon
    norm_b = torch.norm(vector_b) + epsilon
    
    # Calculate cosine of the angle
    cos_theta = dot_product / (norm_a * norm_b)
    
    # Calculate loss as 1 - cos(theta)
    loss = 1 - cos_theta
    
    return loss

def median_sort(arr):
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    else:
        return sorted_arr[mid]
    
# %%
def train(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    #余弦退火
    # 定义参数
    eta_max = 1
    eta_min = 0.001
    T_max = 10  #默认值是10
    
    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device 
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    batch_size = args.batch_size

    dataset = get_dataset(args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    offset_model = nn.Sequential(nn.Linear(3, 1),ScaledTanh(-0.1,0.1)).to(device, dtype=dtype)
    
    model = create_network(args).to(device, dtype=dtype)

    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    accelerator.print("AutoKL size: {:.3f}MB".format(get_weight(first_stage_model)))
    accelerator.print("FM size: {:.3f}MB".format(get_weight(model)))
    accelerator.print("offset model size: {:.10f}MB".format(get_weight(offset_model)))

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    #协同优化
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=0.0)
    
    # if args.use_ema:
    #     optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    data_loader, model, optimizer, scheduler = accelerator.prepare(data_loader, model, optimizer, scheduler)

    exp = args.exp
    parent_dir = "./saved_info/latent_flow/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
    accelerator.print("Exp path:", exp_path)

    if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        ckpt = checkpoint["model_dict"]

        try:
            model.load_state_dict(ckpt)
            print("have module")
        except Exception as e:
            for key in list(ckpt.keys()):
                ckpt[key[7:]] = ckpt.pop(key)
            model.load_state_dict(ckpt)
            print("no module")
        
        
        # load G
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]

        accelerator.print("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint

    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch = int(args.model_ckpt.split("_")[-1][:-4])
        init_epoch = epoch
        model.load_state_dict(checkpoint)
        global_step = 0

        accelerator.print("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    use_label = True if "imagenet" in args.dataset else False
    is_latent_data = True if "latent" in args.dataset else False
    log_steps = 0
    start_time = time()
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    

    # tmux 2:5 blocks tmux 0:10 blocks tmux 3:3z0 tmux 1:10blocks means
    for epoch in range(init_epoch, args.num_epoch + 1):
        z_0_list = []
        z_1_list = []
        z_0_mean_list = []
        for iteration, (x, y) in enumerate(data_loader):
            x_0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            model.zero_grad()
            if is_latent_data:
                z_0 = x_0 * args.scale_factor
            else:
                latent_dis = first_stage_model.encode(x_0).latent_dist
                z_0 = latent_dis.sample().mul_(args.scale_factor) 
            
            z_0_list.append(z_0)    
            z_1 = torch.randn_like(z_0)  
            z_1_list.append(z_1)
        
        st = time()
        print("Initial memory usage:")
        print_memory_usage()
        
        z_1_list = reorder_list_conv(z_0_list, z_1_list, block_size=5)
        
        print("Final memory usage:")
        print_memory_usage()
        et = time()
        print("Epoch %d : reorder time = %3f"%(epoch,et-st))
        for iteration, (x, y) in enumerate(data_loader):
            x_0 = x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            
            sample_num = 1
            loss = 0
            z_0 = z_0_list[iteration]
            z_1 = z_1_list[iteration]

            state = ""
            if state == "discrete":
                #为了防止抽到大量重复值，我们基于隐变量总数*10左右估计得到区域的大小
                z_1 = torch.round(z_1*1e5)/1e5
            
            mode_sample = "uniform"
            if mode_sample == "uniform":
                t = torch.rand((z_0.size(0),), dtype=dtype, device=device)
            elif mode_sample == "logit":
                # 均值就是0.5
                # 假设 t 是你的样本张量
                t = torch.randn((z_0.size(0),), dtype=dtype, device=device)
                # 乘以尺度参数
                t_scaled = t * 1
                # 将 z_scaled 映射到 (0, 1) 区间
                t = torch.sigmoid(t_scaled)       
            elif mode_sample == "mode":     
                s = 0.8
                num_samples = z_0.size(0)
                t = sample_t(s, num_samples)
                
            t = t.view(-1, 1, 1, 1)
            t_origin = t
            z_t = ((1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1)
            offset = False
            if offset == True:
                # 将这两个张量转换为一维张量
                z0_flattened = z_0.reshape(-1, 1)
                z1_flattened = z_1.reshape(-1, 1)
                t = t.expand(batch_size,4,32,32)
                t_flattened = t.reshape(-1, 1)
                
                # 将这两个一维张量拼接在一起
                input = torch.cat((z0_flattened, z1_flattened, t_flattened), dim=1).to('cuda')
                z_t_offset = offset_model(input)
                # cosine_anneal = cosine_annealing_lr(epoch, eta_max, eta_min, T_max)
                cos_2x = 2 * math.cos(pi/40 * epoch + 2*pi) / (np.power(2,(pi/40 * epoch + 2*pi)/(2*pi)))
                z_t = z_t + z_t_offset.reshape(batch_size, 4, 32, 32) * cos_2x
                    
            # 1 is real noise, 0 is real data
            # t = t.view(-1, 1, 1, 1)
            
            # z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
            u = (1 - 1e-5) * z_1 - z_0
            # estimate velocity
            t = t_origin
            v = model(t.squeeze(), z_t, y)
            loss += F.mse_loss(v, u)
            
            
            loss = loss/sample_num
            accelerator.backward(loss)
            optimizer.step()
            model.zero_grad()
            
            #update_ema(ema, model.module)
            global_step += 1
            log_steps += 1
            if iteration % 100 == 0:
                if accelerator.is_main_process:
                    # Measure training speed:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    accelerator.print(
                        "epoch {} iteration{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                            epoch, iteration, loss.item(), steps_per_sec
                        )
                    )
                    
                    print("cosine loss = {}".format(cosine_similarity_loss(u, v)))
                    print("magnitude loss = {}".format(0.1 * F.mse_loss(v, u)))
                    # Reset monitoring variables:
                    log_steps = 0
                    start_time = time()
            # for idx, name in enumerate(model.state_dict().keys(), start=1):
            #     print(f"{idx}. {name}")
        # 每个epoch记录一次损失
        writer.add_scalar('celebA256-loss/train_adm_5blocks_conv_step3_OTCFM_bn112', loss.item(), global_step)
        
        if not args.no_lr_decay:
            scheduler.step()
        
        if accelerator.is_main_process:
            if epoch % args.plot_every == 0:
                with torch.no_grad():
                    rand = torch.randn_like(z_0)[:4] 
                    if state == "discrete":
                        rand = torch.round(rand*1e5)/1e5
                        
                    if y is not None:
                        y = y[:4]  
                        x = x[:4]  # 就是生成前四张图
                    sample_model = partial(model, y=y)
                    # sample_func = lambda t, x: model(t, x, y=y)
                    fake_sample = sample_from_model(sample_model, rand)[-1] 
                    fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                torchvision.utils.save_image(
                    fake_image,
                    os.path.join(exp_path, "image_epoch_{}.png".format(epoch)),
                    normalize=True,
                    value_range=(-1, 1),
                )
                accelerator.print("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    accelerator.print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        # "ema": ema.state_dict(),
                        "model_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }

                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, "model_{}.pth".format(epoch)),
                )
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")

    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        help="model_type",
        choices=[
            "adm",
            "ncsn++",
            "ddpm++",
            "DiT-B/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
        ],
    )
    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )

    # Original ADM
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--use_origin_adm", action="store_true")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")

    args = parser.parse_args()
    train(args)
