import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from easydict import EasyDict as edict
from typing import Tuple
import os # <-- [新增] 导入 os, 准备用于路径
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
import functools
import math
from torch.utils.checkpoint import checkpoint

class Pix2PixDensityGenerator(nn.Module):
    """
    替代 SwinUnetFeatureExtractor 的类。
    使用 Pix2Pix 的 U-Net 架构从 2D 卫星图预测 3D 体素密度。
    
    接口保持一致：
    - 输入: (B, 3, H, W), 范围 [0, 255]
    - 输出: (B, out_channels, H, W), 范围 [0, +inf) (经过 ReLU)
    """
    def __init__(self, img_size, out_channels, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, **kwargs):
        """
        Args:
            img_size (int): 输入/输出图像的分辨率 (例如 256)。
            out_channels (int): 输出通道数 (对应 Z 轴/体素深度, 例如 64)。
            ngf (int): 最后一层卷积的滤波器数量 (控制网络容量)。
            norm_layer: 归一化层类型。
            use_dropout: 是否在解码器中使用 Dropout (Pix2Pix 原文中为 True, 这里默认为 False 以获得确定的密度)。
        """
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels

        # 动态计算下采样层数，确保最底层变为 1x1
        # 例如 256 -> log2(256) = 8 层
        num_downs = int(math.log2(img_size))

        # 构建 U-Net
        self.net = UnetGenerator(input_nc=3, 
                                 output_nc=out_channels, 
                                 num_downs=num_downs, 
                                 ngf=ngf, 
                                 norm_layer=norm_layer, 
                                 use_dropout=use_dropout)

        # 最终激活函数 (保证密度非负)
        self.final_activation = nn.ReLU(inplace=True)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pix2Pix 预处理: 将 [0, 255] 映射到 [-1, 1]
        """
        # 1. [0, 255] -> [0, 1]
        x = x / 255.0
        # 2. [0, 1] -> [-1, 1] (标准 Pix2Pix 输入分布)
        x = (x - 0.5) / 0.5
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 预处理
        x = self.preprocess(x)
        
        # 2. 通过 U-Net
        out = self.net(x)
        
        # 3. 最终激活 (ReLU -> Density)
        # Pix2Pix 原版通常输出 Tanh [-1, 1], 但我们需要密度 [0, inf)
        return self.final_activation(out)


# -------------------------------------------------------------------------
# 以下是经典的 Pix2Pix U-Net 实现 (源自 pytorch-CycleGAN-and-pix2pix)
# -------------------------------------------------------------------------

class UnetGenerator(nn.Module):
    """
    基于 UnetSkipConnectionBlock 构建的 U-Net 生成器。
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        
        # 构造 U-Net 结构，从最内层开始向外递归构建
        
        # 1. 最内层 (Innermost): 不再下采样，没有 skip connection 的子模块
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True) 
        
        # 2. 中间层 (Intermediate): 
        # 添加 num_downs - 5 层 (对于 256x256, num_downs=8, 中间有 3 层)
        for i in range(num_downs - 5): 
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # 3. 逐步上采样减少通道数
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # 4. 最外层 (Outermost): 直接连接输入和输出
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """
    定义 U-Net 的子模块，具有跳跃连接 (Skip Connection)。
    X -------------------identity---------------------- X
      |-- downsampling -- [submodule] -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # 最外层: 输入 -> [submodule] -> 输出
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # 注意：Pix2Pix 原始实现这里有 Tanh，但我们在 Wrapper 类中处理激活，
            # 这里输出原始 Logits，以便后续接 ReLU
            up = [uprelu, upconv] 
            model = down + [submodule] + up
        elif innermost:
            # 最内层: 只是卷积，没有 submodule
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # 中间层: 包含 skip connection
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connection
            # 将输入 x 连接到 submodule 的输出上
            return torch.cat([x, self.model(x)], 1)
        

class ResConv3dBlock(nn.Module):
    """
    一个简单的 Conv3d 残差块 (Conv -> ReLU -> Conv -> + x -> ReLU)
    它保持输入和输出的通道数和分辨率不变。
    """
    def __init__(self, channels: int):
        super().__init__()
        # 使用 1x3x3 卷积核是一种常见的节约显存的技巧 (在Z轴上用 1)
        # 但我们这里用 3x3x3 来充分学习 3D 关系
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 路径
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # 残差连接
        out = out + residual 
        
        # 最后的激活
        return self.relu(out)
# class ResConv3dBlock(nn.Module):
#     """
#     P3D (Pseudo-3D) Residual Block
#     将 3x3x3 卷积分解为:
#     1. Spatial Conv: (1, 3, 3) 处理平面纹理
#     2. Depth Conv:   (3, 1, 1) 处理高度/深度关系
#     """
#     def __init__(self, channels: int):
#         super().__init__()
        
#         # --- 第一层分解 ---
#         # 1. 空间卷积 (S)
#         self.conv1_s = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), 
#                                  stride=1, padding=(0, 1, 1), bias=False)
#         # 2. 深度卷积 (T/D)
#         self.conv1_d = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
#                                  stride=1, padding=(1, 0, 0), bias=False)
#         self.bn1 = nn.InstanceNorm3d(channels) # 推荐用 InstanceNorm 或 GroupNorm 替代 BatchNorm
        
#         # --- 第二层分解 ---
#         # 3. 空间卷积 (S)
#         self.conv2_s = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), 
#                                  stride=1, padding=(0, 1, 1), bias=False)
#         # 4. 深度卷积 (T/D)
#         self.conv2_d = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
#                                  stride=1, padding=(1, 0, 0), bias=False)
#         self.bn2 = nn.InstanceNorm3d(channels)
        
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
        
#         # 第一路径: Spatial -> Depth -> BN -> ReLU
#         out = self.conv1_s(x)
#         out = self.conv1_d(out)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         # 第二路径: Spatial -> Depth -> BN
#         out = self.conv2_s(out)
#         out = self.conv2_d(out)
#         out = self.bn2(out)
        
#         # 残差连接
#         out = out + residual 
        
#         return self.relu(out)


class FeatureRefiner3D(nn.Module):
    """
    [修改版] 3D 特征提炼器
    修改策略:
    1. 降低位置编码权重 (12维)
    2. 提升内容特征权重 (映射到24维)
    3. 总输入维度 36, 隐藏层 64
    """
    def __init__(self, opt, in_channels, pe_channels, out_channels, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.opt = opt
        
        # --- [修改 2] 定义特征维度的变化 ---
        # 原始内容维度: RGB(3) + Density(1) = 4
        self.raw_content_dim = in_channels + 1 
        # 映射后的内容维度: 24
        self.mapped_content_dim = 24 
        
        # 确保 PE 是 12 维 (由外部 opt.PE_channel=2 保证)
        self.pe_dim = pe_channels 
        
        # 拼接后总维度: 24 (Content) + 12 (PE) = 36
        self.total_in_dim = self.mapped_content_dim + self.pe_dim
        
        self.out_dim = out_channels

        # 1. 创建位置编码缓冲
        pe_3d = self.create_positional_encoding(opt) # (C_pe, N, H, W)
        self.register_buffer("pos_encoding", pe_3d, persistent=False)

        # --- [修改 2] 新增特征映射层 (Feature Mapper) ---
        # 作用: 将 (RGB+Density) 从 4 维映射到 24 维
        # 使用 1x1x1 卷积相当于对每个体素做全连接
        self.feature_mapper = nn.Conv3d(self.raw_content_dim, self.mapped_content_dim, kernel_size=1)
        
        # 初始化映射层，保持数值稳定
        nn.init.kaiming_normal_(self.feature_mapper.weight, mode='fan_out', nonlinearity='relu')
        if self.feature_mapper.bias is not None:
            nn.init.constant_(self.feature_mapper.bias, 0)

        # --- [修改 3] 更新 GroupNorm 和 3D CNN ---
        # 输入维度变为 36
        self.input_norm = nn.GroupNorm(num_groups=4, num_channels=self.total_in_dim)

        # 初始卷积: 36 -> 64
        self.initial_conv = nn.Conv3d(self.total_in_dim, hidden_dim, kernel_size=1)
        
        # 中间提炼层
        blocks = [ResConv3dBlock(hidden_dim) for _ in range(num_blocks)]
        self.refiner_blocks = nn.Sequential(*blocks)
        
        # 最终层
        self.final_conv = nn.Conv3d(hidden_dim, self.out_dim, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        
        print(f"FeatureRefiner3D [优化版] 初始化: ")
        print(f"  Mapping: Raw({self.raw_content_dim}) -> Mapped({self.mapped_content_dim})")
        print(f"  Combine: Mapped({self.mapped_content_dim}) + PE({self.pe_dim}) -> Total({self.total_in_dim})")
        print(f"  Network: Input({self.total_in_dim}) -> Hidden({hidden_dim}) -> Output({self.out_dim})")

    def create_positional_encoding(self, opt):
        # 逻辑保持不变，但依赖 opt.arch.gen.PE_channel = 2 来产生 12 维
        depth_channel = opt.arch.gen.depth_arch.output_nc 
        if opt.optim.ground_prior:
            depth_channel = depth_channel + 1
        
        z_ = torch.arange(depth_channel, dtype=torch.float32) / depth_channel
        x_ = torch.arange(opt.data.sat_size[1], dtype=torch.float32) / opt.data.sat_size[1]
        y_ = torch.arange(opt.data.sat_size[0], dtype=torch.float32) / opt.data.sat_size[0]
        
        Z, X, Y = torch.meshgrid(z_, x_, y_, indexing='ij') 
        input_tensor = torch.stack([Z, X, Y], dim=-1)
        
        shape = input_tensor.shape
        freq = 2**torch.arange(opt.arch.gen.PE_channel, dtype=torch.float32) * np.pi
        spectrum = input_tensor[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos], dim=-2)
        input_enc = input_enc.reshape(*shape[:-1], -1)
        
        pos = input_enc.permute(3, 0, 1, 2)
        return pos

    def forward(self, feature_2d, N_depth, density_volume):
        B, C_in, H, W = feature_2d.shape
        
        # 1. 扩展 2D 特征到 3D
        feature_3d = feature_2d.unsqueeze(2).repeat(1, 1, N_depth, 1, 1) # [B, 3, N, H, W]

        # 处理 Density 维度
        if density_volume.dim() == 4:
            density_input = density_volume.unsqueeze(1) # [B, 1, N, H, W]
        else:
            density_input = density_volume
        
        # 2. 处理 PE 深度不匹配 (Padding 逻辑)
        assert N_depth == self.pos_encoding.shape[1]
        current_depth = density_input.shape[2]
        if current_depth != N_depth:
            diff = N_depth - current_depth
            if diff > 0:
                pad_layer = density_input[:, :, :1, :, :]
                pad_block = pad_layer.repeat(1, 1, diff, 1, 1)
                density_input = torch.cat([pad_block, density_input], dim=2)
            elif diff < 0:
                density_input = density_input[:, :, :N_depth, :, :]

        # --- [修改 2 核心逻辑] ---
        
        # A. 拼接原始内容: RGB(3) + Density(1) = 4通道
        # Shape: [B, 4, N, H, W]
        raw_content = torch.cat([feature_3d, density_input], dim=1)
        
        # B. 映射特征: 4 -> 24 通道
        # Shape: [B, 24, N, H, W]
        mapped_content = self.feature_mapper(raw_content)
        mapped_content = F.relu(mapped_content, inplace=True) # 可选：加个激活增加非线性

        # C. 准备 PE
        pe_3d_batched = self.pos_encoding.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        
        # D. 最终拼接: Mapped(24) + PE(12) = 36 通道
        combined_features = torch.cat([mapped_content, pe_3d_batched], dim=1)

        # 3. 归一化与网络前向
        combined_features = self.input_norm(combined_features)
        
        x = self.initial_conv(combined_features)
        x = self.refiner_blocks(x)
        x = self.final_conv(x)

        # 4. 残差连接与输出
        if C_in == self.out_dim:
            x = x + feature_3d

        base_color = feature_3d
        color_delta = torch.tanh(x) * 0.1
        return torch.clamp(base_color + color_delta, 0, 1)
    
class Mlp(nn.Module):
    """ 多层感知机 (MLP) """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -------------------------------------------------------------------
# [新增] 一个简单的残差块
# -------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    一个简单的 Conv2d 残差块 (Conv -> ReLU -> Conv -> + x -> ReLU)
    它保持输入和输出的通道数和分辨率不变。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 路径
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # 残差连接
        out = out + residual 
        
        # 最后的激活
        return self.relu(out)

# -------------------------------------------------------------------
# 文件 2: 渲染器和 MLP (来自您的第二个代码块)
# -------------------------------------------------------------------

def position_produce(opt): 
    depth_channel =  opt.arch.gen.depth_arch.output_nc 
    if  opt.optim.ground_prior:
        depth_channel = depth_channel+1
    z_ = torch.arange(depth_channel, device=opt.device)/depth_channel
    x_ = torch.arange(opt.data.sat_size[1], device=opt.device)/opt.data.sat_size[1]
    y_ = torch.arange(opt.data.sat_size[0], device=opt.device)/opt.data.sat_size[0]
    
    # [修改] 确保在正确的设备上创建
    Z,X,Y = torch.meshgrid(z_,x_,y_, indexing='ij') # 'ij' 匹配 (N_depth, H, W)
    
    input_tensor = torch.stack([Z,X,Y],dim=-1) # [N, H, W, 3]
    pos = positional_encoding(opt,input_tensor) # [N, H, W, C_pe]
    pos = pos.permute(3,0,1,2) # [C_pe, N, H, W]
    return  pos

def positional_encoding(opt,input_tensor): # [...,N]
    shape = input_tensor.shape
    freq = 2**torch.arange(opt.arch.gen.PE_channel,dtype=torch.float32,device=opt.device)*np.pi # [L]
    spectrum = input_tensor[...,None]*freq # [...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [...,N,2,L]
    input_enc = input_enc.reshape(*shape[:-1],-1) # [...,2NL]
    return input_enc


def get_original_coord(opt):
    '''
    pano_direction [X,Y,Z] x right,y up,z out
    '''
    W,H  = opt.data.pano_size
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    if opt.data.dataset in ['CVACT_Shi', 'CVACT', 'CVACThalf']:
        _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude 
    elif opt.data.dataset in ['CVUSA']:
        _theta = (1 - 2 * (_x) / H) * np.pi/4
    # _phi = math.pi* ( 1 -2* (_y)/W ) # longtitude 
    _phi = math.pi*( - 0.5 - 2* (_y)/W )
    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1) 
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1) 
    pano_direction = np.concatenate((axis0, axis1, axis2), axis=2)
    return pano_direction  


def render(opt,color_input,voxel,pano_direction,PE=None):
    '''
    render ground images from satellite images
    
    color_input: B,C_out,N_depth,H_sat,W_sat 提炼后的3D特征体
    voxel: B,N,H_sat,W_sat density of each grid
    pano_direction: pano ray direction  by their definition
    '''
    sat_W,sat_H = opt.data.sat_size
    BS = color_input.size(0)
    ##### get origin, sample point ,depth

    if opt.data.dataset =='CVACT_Shi':
        origin_height=2      ## the height of photo taken in real world scale
        realworld_scale = 30  ## the real world scale corresponding to [-1,1] regular cooridinate
    elif opt.data.dataset == 'CVUSA':
        origin_height=2      
        realworld_scale = 55  
    else:
        assert Exception('Not implement yet')

    assert sat_W==sat_H
    pixel_resolution = realworld_scale/sat_W #### pixel resolution of satellite image in realworld

    if opt.data.sample_total_length:
        sample_total_length = opt.data.sample_total_length
    else: sample_total_length = (int(max(np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(2)**2), \
        np.sqrt((realworld_scale/2)**2+(realworld_scale/2)**2+(opt.data.max_height-origin_height)**2))/pixel_resolution))/(sat_W/2)

    origin_z = torch.ones([BS,1])*(-1+(origin_height/(realworld_scale/2))) ### -1 is the loweast position in regular cooridinate
    if opt.origin_H_W is None: ### origin_H_W is the photo taken space in regular coordinate
        origin_H,origin_w = torch.zeros([BS,1]),torch.zeros([BS,1])  
    else:
        origin_H,origin_w = torch.ones([BS,1])*opt.origin_H_W[0],torch.ones([BS,1])*opt.origin_H_W[1]
    
    # [修改] 确保在正确的设备上创建
    origin = torch.cat([origin_w,origin_z,origin_H],dim=1).to(opt.device)[:,None,None,:]  ## w,z,h
    sample_len = ((torch.arange(opt.data.sample_number, device=opt.device)+1)*(sample_total_length/opt.data.sample_number))
    
    origin = origin[...,None]
    pano_direction_dev = pano_direction.to(opt.device)[...,None] # [修改] 确保在 opt.device 上
    depth = sample_len[None,None,None,None,:]
    sample_point = origin + pano_direction_dev * depth # w,z,h

    if opt.optim.ground_prior:
        # [修改] 确保在正确的设备上创建
        ground_plane = torch.ones(voxel.size(0),1,voxel.size(2),voxel.size(3),device=opt.device)*1000
        voxel = torch.cat([ground_plane, voxel], 1)

    N = voxel.size(1)
    voxel_low = -1
    voxel_max = -1 + opt.data.max_height/(realworld_scale/2)  ### voxel highest space in normal space
    grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]] ### BS,NUM_point,W,H,3 (w,h,z)
    grid[...,2]   = ((grid[...,2]-voxel_low)/(voxel_max-voxel_low))*2-1  ### grid_space change to sample space by scale the z space
    grid = grid.float()  ## [B, 300, 256, 512, 3]
    
    # 检查 color_input 和 voxel 的深度是否匹配
    # (在 ground_prior=True 后, N 可能会改变)
    if color_input.size(2) != N:
        # 我们假设 color_input 已经具有正确的 N 维
        pass

    alpha_grid = torch.nn.functional.grid_sample(voxel.unsqueeze(1), grid, align_corners=False)
    color_grid = torch.nn.functional.grid_sample(color_input, grid, align_corners=False)

    depth_sample = depth.permute(0,1,2,4,3).reshape(1,-1,opt.data.sample_number,1)
    feature_size = color_grid.size(1)
    color_grid = color_grid.permute(0,3,4,2,1).reshape(BS,-1,opt.data.sample_number,feature_size)
    alpha_grid = alpha_grid.permute(0,3,4,2,1).reshape(BS,-1,opt.data.sample_number)
    intv = sample_total_length/opt.data.sample_number
    output = composite(opt, rgb_samples=color_grid,density_samples=alpha_grid,depth_samples=depth_sample,intv = intv)
    output['voxel']  = voxel
    return output

def composite(opt,rgb_samples,density_samples,depth_samples,intv):
    sigma_delta = density_samples*intv # [B,HW,N]
    
    # [修改] 使用 ReLU 确保密度非负 (SAM 解码器末尾已有 ReLU, 此处为双重保险)
    sigma_delta = F.relu(sigma_delta) 
    
    alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=-1).cumsum(dim=-1)) .exp_() # [B,HW,N]
    prob = (T*alpha)[...,None] # [B,HW,N,1]
    
    # 积分
    depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
    rgb = (rgb_samples*prob).sum(dim=2) # [B,HW,3]
    opacity = prob.sum(dim=2) # [B,HW,1]
    
    # Reshape
    H_pano, W_pano = opt.data.pano_size[1], opt.data.pano_size[0]
    depth = depth.permute(0,2,1).view(depth.size(0),-1, H_pano, W_pano)
    rgb = rgb.permute(0,2,1).view(rgb.size(0),-1, H_pano, W_pano)
    opacity = opacity.view(opacity.size(0),1, H_pano, W_pano)
    
    return {'rgb':rgb,'opacity':opacity,'depth':depth}


# (get_sat_ori, render_sat, composite_sat... 在此省略，训练不需要它们)


# -------------------------------------------------------------------
# [新增] 封装所有逻辑的主模型
# -------------------------------------------------------------------

class SatToPanoModel(nn.Module):
    def __init__(self, opt: edict, sam_checkpoint_path: str):
        super().__init__()
        self.opt = opt

        # 1. 初始化密度生成器
        self.density_channels = opt.arch.gen.depth_arch.output_nc
        self.density_generator = Pix2PixDensityGenerator(
            img_size=opt.data.sat_size[0],
            out_channels=self.density_channels,
            ngf=64
        )
        
        # 2. 初始化颜色/特征提炼器
        in_color_feat_dim = 3 
        
        # --- [修改 1] 计算 PE 维度 ---
        # 确保 opt.arch.gen.PE_channel 被设为 2，这里才能得到 12
        # 3 (Z,X,Y) * 2 (sin,cos) * 2 (freqs) = 12
        pe_feat_dim = 3 * 2 * opt.arch.gen.PE_channel 
        
        # --- [修改 3] Hidden Dim 设为 64 ---
        # 使用 opt 中定义的值，或者如果 opt 还是旧的 128，这里强制改为 64
        hidden_dim = getattr(opt, 'mlp_hidden_dim', 64) 
        
        self.color_refiner = FeatureRefiner3D(
                opt=opt,
                in_channels=in_color_feat_dim,
                pe_channels=pe_feat_dim,   # 应当是 12
                out_channels=in_color_feat_dim,
                hidden_dim=hidden_dim,     # 应当是 64
                num_blocks=2
            )
        
        self.pano_direction_numpy = get_original_coord(opt)

    def forward(self, sat_image_255: torch.Tensor) -> dict:
        # Forward 逻辑保持不变，因为 FeatureRefiner3D 内部处理了所有维度变化
        density_voxel = self.density_generator(sat_image_255)
        
        sat_image_01 = sat_image_255 / 255.0
        
        N_depth = self.density_channels
        if self.opt.optim.ground_prior:
            N_depth += 1
        
        color_volume = self.color_refiner(sat_image_01, N_depth, density_voxel)

        B = sat_image_255.size(0)
        pano_direction = torch.from_numpy(self.pano_direction_numpy).unsqueeze(0).repeat(B, 1, 1, 1)

        pano_render_output = render(
                self.opt,
                color_input=color_volume,
                voxel=density_voxel,
                pano_direction=pano_direction
        )

        return {
            'pano_render': pano_render_output,
            'density_voxel': density_voxel,
            'color_volume': color_volume
        }
    
# -------------------------------------------------------------------
# [修改] __main__ 块现在只用于快速测试
# -------------------------------------------------------------------

def _run_model_test():
    """
    一个用于快速测试模型是否可以运行的私有函数。
    """
    
    # --- 1. 设置配置 (opt) ---
    opt=edict()
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据参数
    opt.data = edict()
    opt.data.pano_size = [512, 256] # [W, H]
    opt.data.sat_size = [256, 256]  # [H, W] (假设为正方形)
    opt.data.dataset = 'CVACT_Shi'
    opt.data.max_height = 20
    opt.data.sample_number = 128 # [修改] 减少采样数以加快训练 (300 较慢)
    opt.data.sample_total_length = 1.0 # (来自上一个 main)

    # 架构参数
    opt.arch = edict()
    opt.arch.gen = edict()
    opt.arch.gen.depth_arch = edict()
    opt.arch.gen.depth_arch.output_nc = 64 # [重要] 密度体素的基础深度
    opt.arch.gen.PE_channel = 10           # 位置编码的频率 (L)
    opt.mlp_hidden_dim = 128               # MLP 隐藏层
    
    # 优化/渲染参数
    opt.optim = edict()
    opt.optim.ground_prior = True # [重要] 设为 True 以匹配 MLP 和 render 逻辑

    # 渲染器所需的其他参数 (来自上一个 main)
    opt.origin_H_W = None 

    # --- 2. 设置路径 ---
    # [!! 重要 !!] 
    # 请将 'sam_vit_l_0b3195.pth' 替换为您的 SAM 权重文件的实际路径
    # 下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    # 在 train.py 中, 这个路径将从 opt 传入
    SAM_CHECKPOINT_PATH = "sam_vit_l_0b3195.pth" 

    # --- 3. 实例化模型和优化器 ---
    print(f"正在使用设备: {opt.device}")
    print("正在初始化 SatToPanoModel...")
    
    # 检查 SAM 检查点是否存在, 如果不存在则跳过测试
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"警告: 找不到 SAM 检查点 '{SAM_CHECKPOINT_PATH}'。")
        print("请下载 SAM 权重以运行此测试。正在跳过...")
        return

    model = SatToPanoModel(opt, SAM_CHECKPOINT_PATH).to(opt.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("模型初始化完成。")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params / 1e6:.2f} M")


    # --- 4. 模拟训练步骤 ---
    print("\n--- 开始模拟训练步骤 ---")
    
    BS = 2 # 批处理大小
    
    dummy_sat_image = torch.randint(
        0, 256, 
        (BS, 3, opt.data.sat_size[0], opt.data.sat_size[1]), 
        dtype=torch.float32
    ).to(opt.device)
    
    dummy_gt_pano = torch.rand(
        BS, 3, opt.data.pano_size[1], opt.data.pano_size[0]
    ).to(opt.device)

    model.train()
    render_output = model(dummy_sat_image)
    pred_rgb = render_output['rgb'] 
    loss = F.mse_loss(pred_rgb, dummy_gt_pano)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"训练步骤 1 完成。损失: {loss.item():.6f}")

    # --- 5. 模拟推理和保存 ---
    print("\n--- 开始模拟推理步骤 ---")
    
    model.eval()
    with torch.no_grad():
        render_output = model(dummy_sat_image)
        
    pred_rgb = render_output['rgb']
    pred_opacity = render_output['opacity']

    output_image = torch.cat([
        pred_rgb[0].cpu(), 
        dummy_gt_pano[0].cpu(), 
        pred_opacity[0].cpu().repeat(3,1,1)
    ], dim=2) 
    
    save_path = "training_output_example.png"
    torchvision.utils.save_image(output_image, save_path)
    
    print(f"推理完成。示例图像已保存到: {save_path}")


if __name__ == '__main__':
    # 当 model.py 被直接运行时, 执行此测试
    _run_model_test()
