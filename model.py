import torch.nn as nn
import torch
from pathlib import Path
import torchvision
import timm 
from efficient_kan import KANLinear
from gcn_lib import Grapher, act_layer
import torch.nn.functional as F
import random
import numpy as np
import torch
from functools import lru_cache
from models.vmamba import vanilla_vmamba_tiny
import torch.nn.init as init
import matplotlib.pyplot as plt
import os


class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees = 3):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degrees

        self.beta_weights = nn.Parameter(torch.zeros(degrees + 1, dtype=torch.float32))
        self.basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degrees + 1, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.basis_weights)

    def beta(self, n, m):
        return (
            ((m + n) * (m - n) * n**2) / (m**2 / (4.0 * n**2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def get_basis(self, x, degree):
        p0 = x.new_ones(x.size())
        if degree == 0:
            return p0.unsqueeze(-1)
        p1 = x
        basis = [p0, p1]
        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            basis.append(p2)
            p0, p1 = p1, p2
        return torch.stack(basis, dim=-1)

    def forward(self, x):
        x = torch.tanh(x).contiguous()
        basis = self.get_basis(x, self.degrees)
        y = torch.einsum(
            "b l d, l o d -> b o",
            basis,
            self.basis_weights            
        )
        y = y.view(-1, self.out_channels)
        return y


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 
                              1, 1, bias=False)
        
        self.gate = nn.Conv2d(in_channels, out_channels, 1, 
                              1, 0, bias=False)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.MLP = GRAMLayer(128, 128, 4)
                

    def forward(self, x):
        # 输入: [B, C, H, W]
        conv_out = self.conv(x)  # [B, out_channels, H, W]
        
        # 门控操作
        gate_out = self.gate(x)  # [B, out_channels, H, W]
        gate_out = torch.sigmoid(gate_out)  # 使用Sigmoid函数生成门控值
        
        # 应用门控值
        gated_out = conv_out * gate_out  # [B, out_channels, H, W]
        
        # 批归一化和激活
        gated_out = self.batch_norm(gated_out)
        output = F.gelu(gated_out)
        h = output.size(3)
        output = self.MLP(output.permute(0,2,3,1).reshape(-1,128)).reshape(-1,h,h,128).permute(0,3,1,2)
        
        return output


class upsample(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class patch_wise(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
        )

    def forward(self, x):
        f = self.conv(x)
        w = self.conv_attent(x)
        pred = (f*w).sum(dim=2).sum(dim=2)/w.sum(dim=2).sum(dim=2)
        return pred

    
class netReg(nn.Module): 
    def __init__(self):
        super(netReg, self).__init__()
        self.model_name = Path(__file__).name[:-3]
        self.model_ft = vanilla_vmamba_tiny()
        self.model_ft.load_state_dict(torch.load(open("vssmtiny_dp01_ckpt_epoch_292.pth", "rb"))["model"])
        
        # self.model_ft = vanilla_vmamba_tiny()
        # checkpoint = torch.load(open("vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth", "rb"))
        # new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")}
        # self.model_ft.load_state_dict(new_state_dict, strict=False)
        
        # self.model_ft = vanilla_vmamba_tiny()
        # checkpoint = torch.load(open("vssmtiny_mask_rcnn_swin_fpn_coco_epoch_12.pth", "rb"))
        # new_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")}
        # self.model_ft.load_state_dict(new_state_dict, strict=False)

        self.gate_conv1 = GatedConv2d(96, 128)
        self.gate_conv2 = GatedConv2d(192, 128)
        self.gate_conv3 = GatedConv2d(384, 128)
        self.gate_conv4 = GatedConv2d(768,128)
        self.gate_conv5 = GatedConv2d(768, 128)
        self.global_pool_0 = nn.AdaptiveAvgPool2d(1)
        self.gcn = Grapher(128, 10, 1, conv='mr', act='gelu', norm='batch',
                                          bias=True, stochastic=False, epsilon=0.2, r=1, n=10, drop_path=0.0,
                                          relative_pos=False)
        
        self.conv1 = upsample(96, 128)
        self.conv2 = upsample(192, 128)
        self.conv3 = upsample(384, 128)
        self.conv4 = upsample(768, 128)
        self.conv5 = upsample(768, 128)
        
        self.qa1 = patch_wise(128, 128)
        self.qb1 = patch_wise(128, 128)
        self.qc1 = patch_wise(128, 128)
        self.qd1 = patch_wise(128, 128)
        self.qe1 = patch_wise(128, 128)
        self.qag1 = patch_wise(128, 128)
        self.qbg1 = patch_wise(128, 128)
        self.qcg1 = patch_wise(128, 128)
        self.qdg1 = patch_wise(128, 128)
        self.qeg1 = patch_wise(128, 128)


        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, input):
        b = input.size(0)
        a = self.model_ft.patch_embed(input)
        a_g1 = self.gate_conv1(a.permute(0, 3, 1, 2))
        a1 = self.conv1(a.permute(0, 3, 1, 2))
        a11 = self.global_pool_0(a1).squeeze()
        a11_p = self.qa1(a1) + a11
        ag_11 = self.global_pool_0(a_g1).squeeze()
        ag_11_p = self.qag1(a_g1) + ag_11
        
        a = self.model_ft.layers[0](a)
        b_g1 = self.gate_conv2(a.permute(0, 3, 1, 2))
        b1 = self.conv2(a.permute(0, 3, 1, 2))
        b11 = self.global_pool_0(b1).squeeze()
        b11_p = self.qb1(b1) + b11
        bg_11 = self.global_pool_0(b_g1).squeeze()
        bg_11_p = self.qbg1(b_g1) +bg_11
        
        a = self.model_ft.layers[1](a)
        c_g1 = self.gate_conv3(a.permute(0, 3, 1, 2))
        c1 = self.conv3(a.permute(0, 3, 1, 2))
        c11 = self.global_pool_0(c1).squeeze()
        c11_p = self.qc1(c1) +c11
        cg_11 = self.global_pool_0(c_g1).squeeze()
        cg_11_p = self.qcg1(c_g1) + cg_11
        
        a = self.model_ft.layers[2](a)
        d_g1 = self.gate_conv4(a.permute(0, 3, 1, 2))
        d1 = self.conv4(a.permute(0, 3, 1, 2))
        d11 = self.global_pool_0(d1).squeeze()
        d11_p = self.qd1(d1) + d11
        dg_11 = self.global_pool_0(d_g1).squeeze()
        dg_11_p = self.qdg1(d_g1) + dg_11
        
        a = self.model_ft.layers[3](a)
        e_g1 = self.gate_conv5(a.permute(0, 3, 1, 2))
        e1 = self.conv5(a.permute(0, 3, 1, 2))
        e11 = self.global_pool_0(e1).squeeze()
        e11_p = self.qe1(e1) + e11
        eg_11 = self.global_pool_0(e_g1).squeeze()
        eg_11_p = self.qeg1(e_g1) + eg_11
        
        
        x_1_dim = a11_p.squeeze().unsqueeze(-1)
        # print(x_1_dim.shape)
        x_2_dim = b11_p.squeeze().unsqueeze(-1)
        # print(x_2_dim.shape)
        x_3_dim = c11_p.squeeze().unsqueeze(-1)
        # print(x_3_dim.shape)
        x_4_dim = d11_p.squeeze().unsqueeze(-1)
        # print(x_4_dim.shape)
        x_5_dim = e11_p.squeeze().unsqueeze(-1)
        # print(x_5_dim.shape)
        x_g1_dim = ag_11_p.squeeze().unsqueeze(-1)
        # print(x_1_dim.shape)
        x_g2_dim = bg_11_p.squeeze().unsqueeze(-1)
        # print(x_2_dim.shape)
        x_g3_dim = cg_11_p.squeeze().unsqueeze(-1)
        # print(x_3_dim.shape)
        x_g4_dim = dg_11_p.squeeze().unsqueeze(-1)
        # print(x_g4_dim.shape)
        x_g5_dim = eg_11_p.squeeze().unsqueeze(-1)
        
        x_all = torch.cat((x_1_dim,x_g1_dim,x_2_dim,x_g2_dim,x_3_dim,x_g3_dim,x_4_dim,x_g4_dim,x_5_dim,x_g5_dim), dim = -1)
        
        output = self.gcn(x_all)
        output = self.fc1(output)
        output = self.fc2(output)
        return output
