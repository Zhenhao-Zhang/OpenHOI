import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import time

import torch

from lib.models.mano import build_mano_aa
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    get_object_hand_info, 
    get_valid_mask_bunch, 
    proc_results, 
)
from lib.utils.model_utils import (
    build_refiner, 
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_mpnet, 
    build_pointnetfeat, 
    build_contact_estimator, 
)
from lib.models.object import build_object_model
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.utils.file import (
    make_save_folder, 
    save_video, 
    save_mesh_obj, 
)
from lib.utils.proc import (
    proc_obj_feat_final, 
    proc_cond_contact_estimator, 
    proc_refiner_input, 
)
from lib.utils.visualize import render_videos

from hydra import initialize, compose

import yaml

@hydra.main(version_base=None, config_path="../configs", config_name="config")
@torch.no_grad()

def get_yaml(path):
    with open(path,'r') as file:
        config = yaml.safe_load(file)
    return config

def load_hydra_cfg():
    # 指定你的 configs 目录（相对于当前脚本或项目根）
    with initialize(config_path="../configs"):
        # compose 出与 @hydra.main 相同的 cfg 对象
        cfg = compose(config_name="config")
    return cfg

def create_hoi(text,config,lhand_layer,rhand_layer,refiner,texthom,diffusion,clip_model,mpnet,seq_cvae,pointnet,contact_estimator,object_model):
    #print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
   
    max_nframes = data_config.max_nframes
    #text = config.test_text
    hand_nfeats = config.texthom.hand_nfeats
    obj_nfeats = config.texthom.obj_nfeats

    # lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    # rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()

    # refiner = build_refiner(config, test=True)
    # texthom, diffusion \
    #     = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    # clip_model = load_and_freeze_clip(config.clip.clip_version)
    # clip_model = clip_model.cuda()
    # mpnet = build_mpnet(config)
    # seq_cvae = build_seq_cvae(config, test=True)
    # pointnet = build_pointnetfeat(config, test=True)
    # contact_estimator = build_contact_estimator(config, test=True)
    # object_model = build_object_model(data_config.data_obj_pc_path)
    
    
    is_lhand, is_rhand, \
    obj_pc_org, obj_pc_normal_org, \
    normalized_obj_pc, point_sets, \
    obj_cent, obj_scale, \
    obj_verts, obj_faces, \
    obj_top_idx, obj_pc_top_idx \
        = get_object_hand_info(
            object_model, 
            clip_model, 
            text, 
            data_config.obj_root, 
            data_config,
            mpnet,  
        )
    
    bs, npts = normalized_obj_pc.shape[:2]
    
    enc_text = encoded_text(clip_model, text)
    obj_feat = pointnet(normalized_obj_pc)
    
    batch_num = len(text)//64 + 1
    for batch_idx in range(batch_num):
        ecn_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64]
        is_lhand_batch = is_lhand[batch_idx*64:(batch_idx+1)*64]
        is_rhand_batch = is_rhand[batch_idx*64:(batch_idx+1)*64]
        obj_cent_batch = obj_cent[batch_idx*64:(batch_idx+1)*64]
        obj_scale_batch = obj_scale[batch_idx*64:(batch_idx+1)*64]
        obj_feat_batch = obj_feat[batch_idx*64:(batch_idx+1)*64]
        enc_text_batch = enc_text[batch_idx*64:(batch_idx+1)*64]
        obj_pc_org_batch = obj_pc_org[batch_idx*64:(batch_idx+1)*64]
        obj_pc_normal_org_batch = obj_pc_normal_org[batch_idx*64:(batch_idx+1)*64]
        normalized_obj_pc_batch = normalized_obj_pc[batch_idx*64:(batch_idx+1)*64]
        point_sets_batch = point_sets[batch_idx*64:(batch_idx+1)*64]
        obj_verts_batch = obj_verts[batch_idx*64:(batch_idx+1)*64]
        obj_faces_batch = obj_faces[batch_idx*64:(batch_idx+1)*64]
        
        if dataset_name == "arctic":
            obj_top_idx_batch = obj_top_idx[batch_idx*64:(batch_idx+1)*64]
            obj_pc_top_idx_batch = obj_pc_top_idx[batch_idx*64:(batch_idx+1)*64]
        else:
            obj_top_idx_batch = None
            obj_pc_top_idx_batch = None
            
        duration = seq_cvae.decode(ecn_text_batch)
        duration *= 150
        duration = duration.long()
        valid_mask_lhand, valid_mask_rhand, valid_mask_obj \
            = get_valid_mask_bunch(
                is_lhand_batch, is_rhand_batch, 
                max_nframes, duration
            )
        obj_feat_final, est_contact_map = proc_obj_feat_final(
            contact_estimator,  
            obj_scale_batch, obj_cent_batch, 
            obj_feat_batch, enc_text_batch, npts, 
            config.texthom.use_obj_scale_centroid, 
            config.contact.use_scale, 
            config.texthom.use_contact_feat, 
        )
        coarse_x_lhand, coarse_x_rhand, coarse_x_obj \
            = diffusion.sampling(
                texthom, obj_feat_final, 
                enc_text_batch, max_nframes, 
                hand_nfeats, obj_nfeats, 
                valid_mask_lhand, 
                valid_mask_rhand, 
                valid_mask_obj, 
                device=torch.device("cuda")
            )
        
        if est_contact_map is None:
            condition = proc_cond_contact_estimator(
                obj_scale_batch, obj_feat_batch, enc_text_batch, 
                npts, config.contact.use_scale
            )
            est_contact_map = contact_estimator.decode(condition)
            est_contact_map = (est_contact_map[..., 0] > 0.5).long()
        
        input_lhand, input_rhand, refined_x_obj, \
            = proc_refiner_input(
                coarse_x_lhand, coarse_x_rhand, coarse_x_obj, 
                lhand_layer, rhand_layer, obj_pc_org_batch, obj_pc_normal_org_batch, 
                valid_mask_lhand, valid_mask_rhand, valid_mask_obj, 
                est_contact_map, dataset_name, obj_pc_top_idx=obj_pc_top_idx_batch
            )
        
        refined_x_lhand, refined_x_rhand \
            = refiner(
                input_lhand, input_rhand,  
                valid_mask_lhand=valid_mask_lhand, 
                valid_mask_rhand=valid_mask_rhand, 
            )
    return refined_x_lhand[0],refined_x_rhand[0],refined_x_obj[0]





