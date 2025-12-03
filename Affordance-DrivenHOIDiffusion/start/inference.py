import create_hoi


import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict as edict

import torch
import torch.optim as optim

from lib.models.mano import build_mano_aa, right_hand_mean, left_hand_mean
from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_model_and_diffusion, 
    build_seq_cvae, 
    build_pointnetfeat, 
    build_contact_estimator, 
)




import yaml
from hydra import initialize, compose
from scipy import linalg

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

import pickle
global afford_dict
with open("/home/zhzhang/桌面/Research/LLM-GuidedGraspGeneration/Text2HOI/afford/data_grab.pkl", "rb") as f:
    afford_dict = pickle.load(f)
print(len(afford_dict.keys()))

@hydra.main(version_base=None, config_path="../configs", config_name="config")


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


def main(config):
    print(OmegaConf.to_yaml(config))
    config1=config.copy()
    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name
    

    dataloader = get_dataloader("Motion"+dataset_name, config, data_config)

    pointnet = build_pointnetfeat(config, test=True)

    dump_data = torch.randn([1, 1024, 3]).cuda()
    pointnet(dump_data)

    
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    print(123456,config.clip.clip_version)
    clip_model = clip_model.cuda()

    

    
    nepoch = config.texthom.iteration / (data_config.data_num/data_config.text_num)
    nepoch = int(np.ceil(nepoch / 50.0) * 50)

    # load
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).cuda()

    refiner = build_refiner(config, test=True)
    texthom, diffusion \
        = build_model_and_diffusion(config, lhand_layer, rhand_layer, test=True)
    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()
    mpnet = build_mpnet(config)
    seq_cvae = build_seq_cvae(config, test=True)
    pointnet = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)
    object_model = build_object_model(data_config.data_obj_pc_path)
    fid_sum=0.0
    mpjpe_sum=0.0
    fol_sum=0.0
    diversity_sum=0.0
    diversity_sum_real=0.0
    mmdist_sum=0.0
    count=0
    sum_smooth=0
    cnt_smooth=0
    with tqdm.tqdm(range(1)) as pbar:
        for epoch in pbar:
            for item in dataloader:
                if dataset_name == "arctic":
                    obj_pc_top_idx = item["obj_pc_top_idx"].cuda()
                else:
                    obj_pc_top_idx = None
                x_lhand = item["x_lhand"].cuda()
                x_rhand = item["x_rhand"].cuda()
                x_obj = item["x_obj"].cuda()
                text = item["text"]
                #cov_map
                cov_map = item["cov_map"].cuda()
                
                # cov_map = torch.ones_like(cov_map)
                x_d_lhand ,x_d_rhand,x_d_obj= x_lhand.clone(),x_rhand .clone(),x_obj.clone()
                for i in range(len(x_lhand)):
                    tensorGTlhand, tensorGTrhand, tensorGTobj=x_lhand[i],x_rhand[i],x_obj[i]
                    affordance_map1 = cov_map[i].view(1,1024,1)
                    affordance_map = afford_dict[text[i]][:, :1024, :]
                    print(sum(abs(affordance_map-affordance_map1)))
                    tensorGenlhand, tensorGenrhand, tensorGenobj=create_hoi.create_hoi(text[i],config1,lhand_layer,rhand_layer,refiner,texthom,diffusion,clip_model,mpnet,seq_cvae,pointnet,contact_estimator,object_model,affordance_map)
                    
           
                    
                    

if __name__ == "__main__":
    config = load_hydra_cfg()
    main(config)

