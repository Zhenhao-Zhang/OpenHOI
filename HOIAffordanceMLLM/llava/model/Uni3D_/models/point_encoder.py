import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import logging
# from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
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


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3 512,3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M   12,512,64
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
#12,512,64,3
        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features #12,512,64,6

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 512 64
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 512 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 64
        feature = self.second_conv(feature) # BG 1024 64
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 12 512
        return feature_global.reshape(bs, g, self.encoder_channel)

############################################
# class DGCNN_Propagation(nn.Module):
#     def __init__(self, k = 16):
#         super().__init__()
#         '''
#         K has to be 16
#         '''
#         # print('using group version 2')
#         self.k = k
#         self.knn = KNN(k=k, transpose_mode=False)

#         self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 512),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    )

#         self.layer2 = nn.Sequential(nn.Conv2d(1024, 768, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 768),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    )

#     @staticmethod
#     def fps_downsample(coor, x, num_group):
#         xyz = coor.transpose(1, 2).contiguous() # b, n, 3
#         fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

#         combined_x = torch.cat([coor, x], dim=1)

#         new_combined_x = (
#             pointnet2_utils.gather_operation(
#                 combined_x, fps_idx
#             )
#         )

#         new_coor = new_combined_x[:, :3]
#         new_x = new_combined_x[:, 3:]

#         return new_coor, new_x

#     def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

#         # coor: bs, 3, np, x: bs, c, np

#         k = self.k
#         batch_size = x_k.size(0)
#         num_points_k = x_k.size(2)
#         num_points_q = x_q.size(2)

#         with torch.no_grad():
#             _, idx = self.knn(coor_k, coor_q)  # bs k np
#             assert idx.shape[1] == k
#             idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
#             idx = idx + idx_base
#             idx = idx.view(-1)
#         num_dims = x_k.size(1)
#         x_k = x_k.transpose(2, 1).contiguous()
#         feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
#         feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
#         x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
#         feature = torch.cat((feature - x_q, x_q), dim=1)
#         return feature

#     def forward(self, coor, f, coor_q, f_q):
#         """ coor, f : B 3 G ; B C G
#             coor_q, f_q : B 3 N; B 3 N
#         """
#         # dgcnn upsample
#         f_q = self.get_graph_feature(coor_q, f_q, coor, f)
#         f_q = self.layer1(f_q)
#         f_q = f_q.max(dim=-1, keepdim=False)[0]

#         f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
#         f_q = self.layer2(f_q)
#         f_q = f_q.max(dim=-1, keepdim=False)[0]

#         return f_q
# ########################

class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer):
        super().__init__()
        from easydict import EasyDict
        self.trans_dim = 768#args.pc_feat_dim # 768
        self.embed_dim = 1024#args.embed_dim # 512
        self.group_size = 32#args.group_size # 32
        self.num_group = 512#args.num_group # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)#12,512,4,6

        # define the encoder
        self.encoder_dim =  512#rgs.pc_encoder_dim # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        dr = 0
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(dr) if dr > 0. else nn.Identity()
        self.visual = point_transformer
        
        


    def forward(self, pts, colors):
        # divide the point cloud in the same form. This is important
        
        _, center, features = self.group_divider(pts, colors)
        #print(features.shape)#([12, 512, 32, 6])batch num_group size xyz+color
        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #      batch group encoder_dim
        #print(group_input_tokens.shape)
        group_input_tokens = self.encoder2trans(group_input_tokens)#12,512 768 batch group pc_feat_dim
        #print(group_input_tokens.shape)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  #全局特征
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  #位置
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        # print(x.shape)
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        # print(x.shape)
        x = self.visual.pos_drop(x)
        # print(x.shape)
        # ModuleList not support forward
        features = {}
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
            if i == 3:  
                features['h4'] = x
            elif i == 7:  
                features['h8'] = x
        
        features['h_last'] = x

        h4 = self.visual.norm(features['h4'])
        h8 = self.visual.norm(features['h8'])
        h_last = self.visual.norm(features['h_last'])
    
        h4 = h4[:, 1:, :]
        h8 = h8[:, 1:, :]
        h12 = h_last[:, 1:, :]

        center_level_0 = pts.permute(0,2,1) 
        f_level_0 = torch.cat([center_level_0, center_level_0], 1)

        center_level_1 = fps(pts, 1536).transpose(-1, -2).contiguous()            
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 1024).transpose(-1, -2).contiguous()            
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()                 

        

        # print(h4.shape)
        # print(h8.shape)
        # print(h_last.shape)

        # print(x.shape)
        y = x[:, 1:, :]
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)

        x = self.trans2embed(x)
        # print(y.shape)
        
        return h4,h8,h12,pts,center_level_0,center_level_1,center_level_2,center_level_3