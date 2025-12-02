# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import json
import copy
import time
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from functools import partial
import torch
import transformers
import deepspeed
from llava.constants import IGNORE_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, \
    DEFAULT_PT_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava.model import *
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_point_token, process_pts, load_pts, occlusion, rotation
from llava.train.aff_rea_dataset import ReasonSegDataset
from llava.mm_utils import tokenizer_point_token
import torch.distributed as dist
from enum import Enum

import numpy as np


local_rank = None


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    

import shutil


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_path: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_pt_start_end: bool = field(default=False)
    mm_use_pt_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    prompt_token_num: int = field(default=1)
    with_ape: bool = field(default=True)
    with_local: bool = field(default=True)
    with_global: bool = field(default=True)
    with_color: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    point_folder: Optional[str] = field(default=None)
    sample_points_num: int = field(default=4096)
    occlusion: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    auto_resume:bool=field(default=True)
    resume:str = ""

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_pt_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_POINT_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_POINT_TOKEN, '').strip()
                sentence['value'] = DEFAULT_POINT_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = DEFAULT_POINT_TOKEN
            if data_args.mm_use_pt_start_end:
                replace_token = DEFAULT_PT_START_TOKEN + replace_token + DEFAULT_PT_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_POINT_TOKEN, replace_token)

    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_point: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_point:
        input_ids = torch.stack(
            [tokenizer_point_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # for target in targets:
    #     if 5519 in target and 1822 in target:
    #         if random.random() < 0.5:
    #             loc_start = target.tolist().index(5519)
    #             loc_end = target.tolist().index(1822)
    #             target[:loc_start] = IGNORE_INDEX
    #             target[loc_end + 1:] = IGNORE_INDEX

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, _round in enumerate(rounds):
            if _round == "":
                break

            parts = _round.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_point:
                round_len = len(tokenizer_point_token(_round, tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(_round).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(sources)
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_POINT_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_POINT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_point_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_point_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_point: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_point=has_point)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         pt_tokens = 128 if 'point' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + pt_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'point' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        assert "conversations" in sources
        if 'point' in sources:
            point_file = sources['point']
            point_folder = self.data_args.point_folder
            point = load_pts(os.path.join(point_folder, point_file))
            if self.data_args.occlusion:
                point = process_pts(point, self.data_args)
                point = occlusion(point, self.data_args.sample_points_num, fix=True)
            point = process_pts(point, self.data_args)
            if 'rotation' in sources:
                point[:, :3] = rotation(point[:, :3], sources['rotation'])
            sources = preprocess_multimodal(
                copy.deepcopy([sources["conversations"]]),
                self.data_args)
        else:
            sources = copy.deepcopy([sources["conversations"]])
            point = None

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_point=('point' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['point'] = point
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point' in instances[0]:
            points = [instance['point'] for instance in instances]
            if all(x is not None and x.shape == points[0].shape for x in points):
                batch['points'] = torch.stack(points)
            else:
                batch['points'] = points

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


from torch.utils.tensorboard import SummaryWriter

def train():
    os.makedirs("/root/autodl-tmp/log_dir", exist_ok=True)
    writer = SummaryWriter("/root/autodl-tmp/log_dir")

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    data_args.with_color = model_args.with_color
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[AFF]")
    seg_token_idx = tokenizer("[AFF]", add_special_tokens=False).input_ids[0]

    models_args = {
        "seg_token_idx": seg_token_idx,
        "use_mm_start_end": True,
    }

    bnb_model_from_pretrained_args.update(models_args)
  


    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    model = LISAForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id




    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_r = training_args.lora_r
        if lora_r > 0:

            def find_linear_layers(model, lora_target_modules):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if (
                        isinstance(module, cls)
                        and all(
                            [
                                x not in name
                                for x in [
                                    "visual_model",
                                    "vision_tower",
                                    "mm_projector",
                                    "text_hidden_fcs",
                                ]
                            ]
                        )
                        and any([x in name for x in lora_target_modules])
                    ):
                        lora_module_names.add(name)
                return sorted(list(lora_module_names))
        
        targe_modules = "q_proj,v_proj"
        lora_target_modules = find_linear_layers(
            model, targe_modules.split(",")
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )



        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    



    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    # vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_pt_start_end = data_args.mm_use_pt_start_end = model_args.mm_use_pt_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_pt_start_end = model_args.mm_use_pt_start_end
    model.config.mm_use_pt_patch_token = model_args.mm_use_pt_patch_token
    model.config.with_color = model_args.with_color
    model.config.sample_points_num = data_args.sample_points_num
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    model.resize_token_embeddings(len(tokenizer))
    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "text_hidden_fcs",
                        "projection","Geometry_Correlation","propagation_2" ,"propagation_1","propagation_0",
                         "dgcnn_pro_1","dgcnn_pro_0","decoder"     ]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    world_size = torch.cuda.device_count()
    train_dataset = ReasonSegDataset( 
                 1,
                 samples_per_epoch=2000 * 2 * 1 * 10,
                 exclude_val=False,
                 reason_seg_data="/root/autodl-tmp/affdata/point_train_all.txt",
                 run_type = "train",
                 explanatory=-1,
                 json_path = "/root/autodl-tmp/affdata/json_train_all.txt"
                 )
    
    test_dataset = ReasonSegDataset( 
                 1,
                 exclude_val=False,
                 reason_seg_data="/root/autodl-tmp/affdata/point_test_all.txt",
                 run_type = "test",
                 explanatory=-1,
                 json_path = "/root/autodl-tmp/affdata/json_test_all.txt"
                 )
    print(f"Training with {len(train_dataset)} examples.")


    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 10,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-4,
                "weight_decay": 0.0,
                "betas": (0.9, 0.95),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 10 * 2000,
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-4,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        # "fp16": {
        #     "enabled": model_args.precision == "fp16",
        # },
        # "bf16": {
        #     "enabled": args.precision == "bf16",
        # },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_mm_start_end=model.config.mm_use_pt_start_end,
            local_rank=local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    # if training_args.auto_resume and len(training_args.resume) == 0:
    #     resume = os.path.join("/root/autodl-tmp/log_dir", "ckpt_model")
    #     if os.path.exists(resume):
    #         training_args.resume = resume

    # if training_args.resume:
    #     load_path, client_state = model_engine.load_checkpoint(training_args.resume)
    #     with open(os.path.join(training_args.resume, "latest"), "r") as f:
    #         ckpt_dir = f.readlines()[0].strip()
    #     start_epoch = (
    #         int(ckpt_dir.replace("global_step", "")) // 2000
    #     )
    #     print(
    #         "resume training from {}, start from epoch {}".format(
    #             training_args.resume, start_epoch)
    #     )


    

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                use_mm_start_end=model.config.mm_use_pt_start_end,
                local_rank=local_rank,
            ),
        )

    train_iter = iter(train_loader)
    test_leng = len(test_dataset)
    best_auc = 0.0
    for epoch in range(0, 10):
        # train for one epoch
        train_iter = train_epoch(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
        )

        auc = validate(test_loader, model_engine,epoch, writer,test_leng)
        is_best = auc > best_auc
        best_auc = max(auc,best_auc)

        if is_best:
            save_dir = os.path.join("/root/autodl-tmp/log_dir", "ckpt_model")
            torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        "/root/autodl-tmp/log_dir",
                        "meta_log_AUC{:.3f}.pth".format(
                            best_auc
                        ),
                    ),
                )
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)






    


if __name__ == "__main__":
    train()



def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    point_list = []
    conversation_list = []
    aff_pred_list = []
    questions_list = []
    offset_list = [0]
    cnt = 0
    affordance_label_list = []
    logist_label_list =[]

    for (points,conversations,aff_preds,questions,_,affordance_label,logist_label) in batch:
        points = torch.tensor(points)
        points = points.to(torch.float32) 
        point_list.append(points)
        conversation_list.extend(conversations)
        aff_pred_list.append(aff_preds)
        questions_list.append(questions)
        cnt += len(conversations)
        offset_list.append(cnt)
        affordance_label_list.append(torch.from_numpy(affordance_label))
        logist_label_list.append(logist_label)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_POINT_TOKEN
            replace_token = (
                    DEFAULT_PT_START_TOKEN + replace_token + DEFAULT_PT_END_TOKEN
                )
            
            
            conversation_list[i] = conversation_list[i].replace(
                    DEFAULT_POINT_TOKEN, replace_token
                )
    input_ids = [
        tokenizer_point_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
                ]
    
   
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # input_ids = torch.stack(input_ids, dim=0)


    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()


    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou, tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    truncate_len = tokenizer.model_max_length - 255

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]
    return{
            "points":torch.stack(point_list, dim=0),
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            
            "offset": torch.LongTensor(offset_list),
            "aff_label":torch.stack(affordance_label_list, dim=0),
            "logist_label":logist_label_list
            
            
           


        }


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def train_epoch(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
):

    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    

    progress = ProgressMeter(
        2000,
        [
            batch_time,
            losses,
            ce_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    for global_step in range(2000):
        for i in range(10):
            try:
                input_dict = next(train_iter)
                
                 
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            input_dict = dict_to_cuda(input_dict)
            points = input_dict["points"]
            input_ids = input_dict["input_ids"]
            labels = input_dict["labels"]
            attention_masks = input_dict["attention_masks"]
            offset = input_dict["offset"]
            aff_label = input_dict["aff_label"]
            logist_label = input_dict["logist_label"]
            
            data_time.update(time.time() - end)
            loss,pred_affordance,aff_targets = model(points=points,input_ids=input_ids,labels=labels,attention_masks = attention_masks,offset = offset,aff_label = aff_label,logist_label=logist_label )

            losses.update(loss.item(), input_dict["points"].size(0))
            model.backward(loss)
            model.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % 10 == 0:
            progress.display(global_step + 1)
            writer.add_scalar("train/loss", losses.avg, global_step)
            writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
            writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

        batch_time.reset()
        data_time.reset()
        losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", curr_lr[0], global_step)


    return train_iter

import tqdm
def validate(val_loader, model_engine, epoch, writer,length):
    AUC_meter = AverageMeter("AUC", ":6.3f", Summary.SUM)
    SIM__meter = AverageMeter(" SIM", ":6.3f", Summary.SUM)
    iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    model_engine.eval()

    pr_aff, gt_aff = [], []
    aff_preds = torch.zeros((length, 2048, 1))
    aff_targets = torch.zeros((length, 2048, 1))

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        points = input_dict["points"]
        input_ids = input_dict["input_ids"]
        labels = input_dict["labels"]
        attention_masks = input_dict["attention_masks"]
        offset = input_dict["offset"]
        aff_label = input_dict["aff_label"]
        logist_label = input_dict["logist_label"]
        

        with torch.no_grad():            
            loss_ca,pred_affordance,aff_targets = model_engine(points=points,input_ids=input_ids,labels=labels,attention_masks = attention_masks,offset = offset,aff_label = aff_label,logist_label=logist_label )
            aff_targets = aff_targets.unsqueeze(dim=-1)
            pred_affordance =  torch.cat(pred_affordance, dim=0)
            pr_aff.append(pred_affordance)
            gt_aff.append(aff_targets)

    aff_preds = torch.cat(pr_aff, 0)
    aff_targets = torch.cat(gt_aff, 0)

    AUC_, IOU_, SIM_ = evaluate(aff_preds, aff_targets)
    AUC_meter.update( AUC_)
    SIM__meter.update(SIM_)
    iou_meter.update(IOU_)
    AUC = AUC_meter.avg
    SIM = SIM__meter.avg
    IOU = iou_meter.avg


    AUC_meter.all_reduce()
    SIM__meter.all_reduce()
    iou_meter.all_reduce()

    writer.add_scalar("AUC/val",AUC, epoch)
    writer.add_scalar("SIM/val", SIM, epoch)
    writer.add_scalar("IOU/val", IOU, epoch)

    print("AUC: {:.4f}, SIM: {:.4f}, IOU:{:.4f}".format(AUC_meter.avg, SIM__meter.avg,iou_meter.avg))
    return AUC_meter.avg




from sklearn.metrics import roc_auc_score
def evaluating(pred, label):

    mae = torch.sum(torch.abs(pred-label), dim=(0,1))
    points_num = pred.shape[0] * pred.shape[1]

    return mae, points_num

def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld
    
def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def evaluate(aff_pred, aff_gt):

    '''
    affordance:[B, 2048, 1]
    '''
    # dist_matrix = np.load('smpl_models/smpl_neutral_geodesic_dist.npy')
    # dist_matrix = torch.tensor(dist_matrix)

   
    aff_pred = aff_pred.cpu().detach().numpy()
    aff_gt = aff_gt.cpu().detach().numpy()

    AUC_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))
    IOU_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))

    SIM_matrix = np.zeros(aff_gt.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    num = aff_gt.shape[0]

    for b in range(num):
        #sim
        SIM_matrix[b] = SIM(aff_pred[b], aff_gt[b])

        #AUC_IOU
        aff_t_true = (aff_gt[b] >= 0.5).astype(int)
        aff_p_score = aff_pred[b]

        if np.sum(aff_t_true) == 0:
            AUC_aff[b] = np.nan
            IOU_aff[b] = np.nan
        else:
            try:
                auc_aff = roc_auc_score(aff_t_true, aff_p_score)
                AUC_aff[b] = auc_aff
            except ValueError:
                #print(pts_path[b])
                AUC_aff[b] = np.nan

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (aff_p_score >= thre).astype(int)
                intersect = np.sum(p_mask & aff_t_true)
                union = np.sum(p_mask | aff_t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU_aff[b] = aiou

    AUC_ = np.nanmean(AUC_aff)
    IOU_ = np.nanmean(IOU_aff)
    SIM_ = np.mean(SIM_matrix)

    return AUC_, IOU_, SIM_