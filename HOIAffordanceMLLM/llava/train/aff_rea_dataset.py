import numpy as np
import torch
import torch.nn.functional as F

from llava import conversation as conversation_lib
import glob
import json
import os
import random
from torch.utils.data import DataLoader
DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_PATCH_TOKEN = "<pt_patch>"
DEFAULT_PT_START_TOKEN = "<pt_start>"
DEFAULT_PT_END_TOKEN = "<pt_end>"



SHORT_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_POINT_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_POINT_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_POINT_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_POINT_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

def get_info_from_json(json_path):
        json_path = os.path.normpath(json_path).replace('\\', '/')
        try:
            with open(json_path, "r") as r:
                anno = json.loads(r.read())
        except:
            with open(json_path, "r", encoding="cp1252") as r:
                anno = json.loads(r.read())
        comments = anno["questions"]
        aff_type = anno["affordance_type"]
        return comments,aff_type

def read_text_file(file_path):
    lines_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines_list.append(line.strip())
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return lines_list

def pc_normalize(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


class ReasonSegDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 samples_per_epoch=None,
                 num_classes_per_sample: int = 3,
                 exclude_val=False,
                 reason_seg_data="Affordance_Reasoning_dataset",
                 run_type = "train",
                 explanatory=-1,
                 json_path = "affordance_json/json_all.txt"
                 ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.tokenizer = tokenizer
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        number_dict = {'Earphone': 0, 'Bag': 0, 'Chair': 0, 'Refrigerator': 0, 'Knife': 0, 'Dishwasher': 0, 'Keyboard': 0, 'Scissors': 0, 'Table': 0, 
            'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Microwave': 0, 'Display': 0, 'TrashCan': 0, 'Hat': 0, 'Clock': 0, 
            'Door': 0, 'Mug': 0, 'Faucet': 0, 'Vase': 0, 'Laptop': 0, 'Bed': 0}
        
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'warpgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab',"pull"]
        # if self.run_type == 'train':
        self.point_files, self.number_dict = self.read_file(self.reason_seg_data, number_dict)
        if 'test' in self.number_dict:
            del number_dict['test']
        self.object_list = list(number_dict.keys())
        self.object_train_split = {}
        start_index = 0
        for obj_ in self.object_list:
            temp_split = [start_index, start_index + self.number_dict[obj_]]
            self.object_train_split[obj_] = temp_split
            start_index += self.number_dict[obj_]

        self.all_datasets = []
        self.type = run_type
    
        self.json = read_text_file(json_path)

        

        # if explanatory != -1:
        #     self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
        #     self.affordance_to_explanation = {}
        #     with open(
        #         os.path.join(  "explanatory", "train.json",)) as f:
        #         items = json.load(f)
        #     for item in items:
        #         Point_ID = item[""]
        #         self.Point_to_explanation[Point_ID] = {
        #             "query": item["query"],
        #             "outputs": item["outputs"],
        #         }

        #     print("len(self.img_to_explanation): ", len(self.img_to_explanation))
        

    def __len__(self):
        if  self.samples_per_epoch != None:
            return self.samples_per_epoch
        else:
            return len(self.json)
    

    def __getitem__(self, idx):
        
        idx = random.randint(0, len(self.json) - 1)

        json_path = self.json[idx]
        parts = json_path.split("/")
        file_name = parts[-1].split("_")
        affordance_type = file_name[0]
        affordance_type = affordance_type[11:]
        # print(affordance_type)
        object_name = file_name[2]
        id = file_name[-1].split(".")[0]
        
        range_ = self.object_train_split[object_name]
        point_sample_idx = random.sample(range(range_[0],range_[1]), 1)
        point_path = 'affdata/'+self.type+'/point_'+object_name+'_'+id+'.txt'
        pa = '/root/tmp/'
        # for id_x in point_sample_idx:
                # point_path = self.point_files[id_x]
        Points, affordance_label = self.extract_point_file(pa+point_path)
        Points = pc_normalize(Points)
        Points = Points.transpose()
        affordance_label, affordance_index = self.get_affordance_label(affordance_type, affordance_label)
        
        ori_size = affordance_label.shape
        logist_label = affordance_index
        
        
        # if self.explanatory != -1 and Point_ID in self.Point_to_explanation:
        #     if random.random() < self.explanatory:
        #         choice = 2
        #     else:
        #         choice = random.randint(0, 1)
        choice = random.randint(0, 1)




        sents, aff_type = get_info_from_json(pa+json_path)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        questions = []
        answers = []    

        for text in sampled_sents:
            question_template = random.choice(self.long_question_list)
            questions.append(question_template.format(sent=text))


        # add explanation if applicable
            if self.explanatory != -1:
                print("1")

            # if self.explanatory != -1 and Point_ID in self.Point_to_explanation:
            #     if choice == 0:  # [SEG] token
            #         answers.append(random.choice(self.answer_list))
            #     elif choice == 1:  # [SEG] token + text answer
            #         answer = self.Point_to_explanation[Point_name]["outputs"]
            #         answer = random.choice(self.answer_list) + " {}".format(answer)
            #         texts[-1] = (
            #             DEFAULT_POINT_TOKEN
            #             + "\n"
            #             + text
            #             + " {}".format(random.choice(self.explanatory_question_list))
            #         )
            #         answers.append(answer)
            #     elif choice == 2:  # vanilla text answer
            #         answer = self.Point_to_explanation[Point_name]["outputs"]
            #         questions[-1] = DEFAULT_POINT_TOKEN + "\n" + text
            #         answers.append(answer)
            #     else:
            #         raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1
        
            aff_pred = torch.rand(0, *ori_size)
            
            
            
            
           


            return (
            Points,
            conversations,
            aff_pred,
            questions,
            sampled_sents,
            affordance_label,
            logist_label
        )



    





    def get_affordance_label(self, affordance_type, label):

        affordance_type = affordance_type.replace('\\', '/')
        affordance_type = affordance_type.replace('json\\', 'json/')
        
        index = self.affordance_label_list.index(affordance_type)

        label = label[:, index]
        
        return label, index
    


 

    def extract_point_file(self, path):
        with open(path,'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                parts = file.split('/')
                parts = parts[-1].split('_')
                if number_dict != None:
                    object_ = parts[-2]
                    number_dict[object_] +=1
                file_list.append(file)

            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
