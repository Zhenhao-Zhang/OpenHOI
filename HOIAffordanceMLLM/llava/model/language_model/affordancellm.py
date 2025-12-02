#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import argparse
import os
import shutil
import sys
import time
from functools import partial
from knn_cuda import KNN
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
# from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from llava import conversation as conversation_lib

from llava.model.Uni3D.models import uni3d as modelss
#################################

from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)


import torch
import torch.nn.functional as F
from time import time
import numpy as np

class L_ca(nn.Module):
    def __init__(self):
        super(L_ca, self).__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):

        temp1 = -(1-self.alpha)*torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -self.alpha*torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)
        return CELoss+1.0*DICELoss

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = (torch.ones(B, N).to(device) * 1e10)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm_Atten(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, key_value):
        return self.fn(self.norm(x), self.norm(key_value))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, self.inner_dim*2, bias = False)

    def forward(self, query, key_value):

        B = query.size(0)
        q = self.to_q(query).view(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)            #b n (h d)

        kv = self.to_kv(key_value).chunk(2, dim = -1)       
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_Atten(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, key_value):
        for attn, ff in self.layers:
            x = attn(x, key_value) + x
            x = ff(x) + x
        return x

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss
#################
class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(1536, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1536, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q
    
    ################################################
class Curvature_guided_Geometric_Correlation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        self.f_m1 = Cross_Attention(dim = input_dim, heads = 8, dropout = 0.3, dim_head = 64)
        self.f_m2 = Cross_Attention(dim = input_dim, heads = 8, dropout = 0.3, dim_head = 64)

        self.fusion_hm = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.fusion_obj = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.affordance = Transformer(dim = input_dim, depth = 1, heads = 8, mlp_dim = 512, dropout = 0.3, dim_head = 64)
        self.contact = Transformer(dim = input_dim, depth = 1, heads = 8, mlp_dim = 512, dropout = 0.3, dim_head = 64)

    def forward(self, F_o_, T_o_):

        conditional_aff = T_o_.unsqueeze(dim=1)

        phi_a = self.affordance(F_o_, conditional_aff)

        return phi_a
    
def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss
    
class Decoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        
        self.aff_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            SwapAxes(),
            nn.BatchNorm1d(feat_dim//6),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 1)
        )

        self.contact_up_fine = nn.Linear(1723, 6890)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi_a):

        B = phi_a.size(0)
        affordance = self.aff_head(phi_a)                                  
        affordance = self.sigmoid(affordance)

        return affordance





class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # if not hasattr(self.config, "train_mask_decoder"):
        #     self.config.train_mask_decoder = kwargs["train_mask_decoder"]
        #     self.config.out_dim = kwargs["out_dim"]
        #     self.vision_pretrained = kwargs.get("vision_pretrained", None)
        # else:
            # self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lisa_modules(self.config)
        

    def initialize_lisa_modules(self, config):
        self.point_model = modelss.create_uni3d()
        

        checkpoint = torch.load("/root/autodl-tmp/model.pt", map_location="cpu")
        # logging.info('loaded checkpoint {}'.format(args.ckpt_path))
        sd = checkpoint['module']
        distributed = False
        if not distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.point_model.load_state_dict(sd)

            # 遍历模型的所有参数和名称
        for name, param in self.point_model.named_parameters():            
            # if name.startswith('point_encoder.visual.blocks.10') or name.startswith('point_encoder.visual.blocks.11'):  
            #     param.requires_grad = True
            # else:
            param.requires_grad = False

        print("using Uni3D as the point backbone!")
        self.point_model = self.point_model



    # Projection layer
        in_dim = config.hidden_size
        out_dim = 256#
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):

        # if not hasattr(config, "train_mask_decoder"):
        #     config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
           
        #     self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        #     self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        #     self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        # else:
        config.mm_vision_tower = config.mm_vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.projection = nn.Sequential(
                            nn.Linear(256, 512),   
                            nn.ReLU(),           
                            nn.Linear(512, 512),  
                            nn.ReLU(),             
                                        
                        )
        

        self.Geometry_Correlation = Curvature_guided_Geometric_Correlation(512)
        self.propagation_2 = PointNetFeaturePropagation(in_channel= 768+ 3, mlp = [768, 768])
        self.propagation_1= PointNetFeaturePropagation(in_channel= 768 + 3, mlp = [768 , 768])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= 768 + 3+3, mlp = [768, 512])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)
        self.decoder = Decoder(512)
        self.loss_ca = L_ca()
        self.post_init()

    def get_visual_embs(self, point):
        points = point.transpose(1,2)
        rgb = torch.full_like(points, 0.4)
        points = torch.cat((points, rgb),dim=-1)
        h4,h8,h12,pts,center_level_0,center_level_1,center_level_2,center_level_3= self.model.point_model.encode_pc(points)

        h4 = h4.permute(0,2,1)
        h8 = h8.permute(0,2,1)
        h12=h12.permute(0,2,1)


        f_level_1 = center_level_1
        f_level_2 = center_level_2
         # init the feature by 3nn propagation
        f_level_3 = h12
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, h8)#1024 feature
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, h4)#1536
        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, points.transpose(1,2), f_level_1)

        



        return f_level_0.transpose(1,2)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        points:torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.Tensor,
        offset: torch.LongTensor,
        aff_label:torch.FloatTensor,
        logist_label:torch.FloatTensor,
        **kwargs,
    ):

        Point_embeddings = self.get_visual_embs(points)
        # print(Point_embeddings.shape)
        batch_size = Point_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        points = points.transpose(1,2)
        rgb = torch.full_like(points, 0.4)
        points = torch.cat((points, rgb),dim=-1)
        points_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            points_i = (
                points[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1)
                .contiguous()
            )
            points_list.append(points_i)
        points_input = torch.cat(points_list, dim=0)
      
        output = super().forward(
                points=points_input,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
        output_hidden_states = output.hidden_states
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        sequence_length = last_hidden_state.size(1)
        padding_length = max(0, sequence_length - seg_token_mask.size(1))
        seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], padding_length), dtype=torch.bool).cuda(), seg_token_mask],dim=1,
                                    )
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []

        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        # print("_-----------------------------")
        # print(len(pred_embeddings))
        pred_affordance =[]
        loss_a = 0
        for i in range(len(pred_embeddings)):
            hseg = self.projection(pred_embeddings[i])
            # print(hseg.shape)
            phi_a = self.Geometry_Correlation(Point_embeddings[i].unsqueeze(0), hseg)  # phi_a[16, 2048, 512]

            affordance = self.decoder(phi_a)               # affordance[16, 2048, 1]
            # print( affordance.shape)
            pred_affordance.append(affordance)
            loss_a += self.loss_ca(affordance,aff_label[i].unsqueeze(-1))
        loss_a = loss_a / len(pred_embeddings)
        model_output = output
        ce_loss = model_output.loss
        # mask_bce_loss = 0
        # mask_dice_loss = 0
        # num_masks = 0
        # for batch_idx in range(len(pred_affordance)):
        #     pred_mask = pred_affordance[batch_idx]
        #     gt_mask = aff_label[batch_idx]
        #     mask_bce_loss += (
        #         sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
        #         * gt_mask.shape[0]
        #     )
        #     mask_dice_loss += (
        #         dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
        #         * gt_mask.shape[0]
        #     )
        #     num_masks += gt_mask.shape[0]
        # mask_bce_loss = 0.5 * mask_bce_loss / (num_masks + 1e-8)
        # mask_dice_loss = 2 * mask_dice_loss / (num_masks + 1e-8)
        # mask_loss = mask_bce_loss + mask_dice_loss
        loss = ce_loss +loss_a
            
        return loss,pred_affordance,aff_label



