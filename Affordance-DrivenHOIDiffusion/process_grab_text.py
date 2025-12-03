import os
import os.path as osp

import glob
import numpy as np
import trimesh
import tqdm
import pickle
import time
import json
from collections import Counter

import torch

from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.utils.frame import align_frame
from lib.utils.file import load_config, read_json
from lib.utils.rot import axis_angle_to_rotmat
from lib.utils.proc import (
    proc_torch_cuda, 
    proc_numpy, 
    get_hand_org, 
    get_contact_info, 
    transform_hand_to_xdata, 
    transform_xdata_to_joints, 
    transform_obj_to_xdata, 
    farthest_point_sample, 
)
from lib.utils.proc_grab import process_text
from constants.grab_constants import (
    grab_obj_name, object_proc, motion_proc, 
    present_participle, third_verb, passive_verb, 
)

def preprocessing_text():
    grab_config = load_config("/home/zhzhang/桌面/Grasp/Text2HOI/configs/dataset/grab.yaml")
    print(1,grab_config)
    data_root = grab_config.root
    data_root = '/home/zhzhang/桌面/Grasp/Text2HOI/data/grab'
    print(data_root)
    text_json = grab_config.text_json
    
    data_list = glob.glob(osp.join(data_root, "data.npz"))
    data_list.sort()
    print(2,data_list)

    action_list = []

    for data_path in tqdm.tqdm(data_list):
        data = np.load(data_path, allow_pickle=True)
        print(data[0])
        motion = data["motion_intent"].item()
        if motion in motion_proc:
            motion = motion_proc[motion]
        obj_name = data["obj_name"].item()
        if obj_name in object_proc:
            obj_name = object_proc[obj_name]
        action_list.append(motion + "," + obj_name)
    
    action_list = np.unique(action_list)
    
    text_description = {}
    print(action_list)
    for action in action_list:
        print(action)
        action_v, action_o = action.split(",")[0], action.split(",")[1]
        
        text_left = f"{action_v} {action_o} with left hand.".capitalize()
        text_right = f"{action_v} {action_o} with right hand.".capitalize()
        text_both = f"{action_v} {action_o} with both hands.".capitalize()
        
        action_ving = present_participle[action_v]
        text_left1 = f"{action_ving} {action_o} with left hand.".capitalize()
        text_right1 = f"{action_ving} {action_o} with right hand.".capitalize()
        text_both1 = f"{action_ving} {action_o} with both hands.".capitalize()

        action_3rd_v = third_verb[action_v]
        text_left2 = f"Left hand {action_3rd_v} {action_o}."
        text_right2 = f"Right hand {action_3rd_v} {action_o}."
        text_both2 = f"Both hands {action_v} {action_o}."

        action_passive = passive_verb[action_v]
        text_left3 = f"{action_o} {action_passive} with left hand.".capitalize()
        text_right3 = f"{action_o} {action_passive} with right hand.".capitalize()
        text_both3 = f"{action_o} {action_passive} with both hands.".capitalize()

        text_description[text_left] = [text_left, text_left1, text_left2, text_left3]
        text_description[text_right] = [text_right, text_right1, text_right2, text_right3]
        text_description[text_both] = [text_both, text_both1, text_both2, text_both3]
    '''
    with open(text_json, "w") as f:
        json.dump(text_description, f)
    '''
def preprocessing_balance_weights():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_path = grab_config.data_path
    balance_weights_path = grab_config.balance_weights_path
    t2c_json_path = grab_config.t2c_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        proc_obj_name = data["proc_obj_name"]

    text_list = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], proc_obj_name[i], 
            is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        text_list.append(text_key)
    
    text_counter = Counter(text_list)
    text_dict = dict(text_counter)
    text_prob = {k:1/v for k, v in text_dict.items()}
    balance_weights = [text_prob[text] for text in text_list]
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)
    with open(t2c_json_path, "w") as f:
        json.dump(text_dict, f)
        
def preprocessing_text2length():
    grab_config = load_config("configs/dataset/grab.yaml")
    data_path = grab_config.data_path
    t2l_json_path = grab_config.t2l_json
    
    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        proc_obj_name = data["proc_obj_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i], proc_obj_name[i], 
            is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open(t2l_json_path, "w") as f:
        json.dump(text_dict, f)

preprocessing_text()