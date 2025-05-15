import os
import copy, glob
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import ast

import torch
import time
import random
import cv2

import transformers
import tokenizers
import numpy as np

from ola.constants import IGNORE_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ola.train.ola_trainer import OlaTrainer

from ola import conversation as conversation_lib
from ola.model import *
from ola.datasets.preprocess import tokenizer_speech_token


from PIL import Image, TarIO, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Truncated File Read
Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning
ImageFile.MAX_IMAGE_PIXELS = None

from ola.mm_utils import process_anyres_video, process_anyres_highres_image

from safetensors.torch import load_file as safetensor_load_file
from transformers import AutoConfig


from torch.utils.data import Dataset
from packaging import version
import io, base64, math, pickle
import whisper
import librosa


local_rank = None
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_safetensor_path: Optional[str] = field(default=None)
    resume_from: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    s2s: bool = field(default=False)
    speech_audio: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_speech_adapter: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    speech_encoder: Optional[str] = field(default=None)
    music_encoder: Optional[str] = field(default=None)
    fix_speech_encoder: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_projector_type: Optional[str] = field(default='none')
    speech_encoder_type: Optional[str] = field(default='none')
    speech_encoder_config: Optional[str] = field(default='')
    speech_encoder_ds_rate: Optional[int] = field(default=10)
    speech_encoder_hidden_size: Optional[int] = field(default=1280)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    freeze_speech_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    freeze_mm_vision_tower: bool = field(default=False)
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
    lora_enable: bool = field(default=False)
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    speech_projector_lr: Optional[float] = None
    mm_speech_encoder_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    min_lr_ratio: float = field(default=0.0)
    sample_independently: bool = field(default=False)
    do_resize: bool = field(default=False)
    do_center_crop: bool = field(default=False)


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
    multimodal_keywords = ['speech_projector', 'speech_encoder', 'mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_speech_adapter", False):
        # Only save Adapter
        keys_to_match = ['speech_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                speech_projector_folder = os.path.join(parent_folder, "speech_projector")
                os.makedirs(speech_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(speech_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'speech_projector.bin'))
        return
    elif getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(trainer.args, "use_im_start_end", False):
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

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence['value'] and DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SPEECH_TOKEN, '').strip()
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            elif DEFAULT_SPEECH_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SPEECH_TOKEN, '').strip()
                sentence['value'] = DEFAULT_SPEECH_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            elif DEFAULT_IMAGE_TOKEN in sentence['value']:
                num_image = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = ( DEFAULT_IMAGE_TOKEN + '\n' ) * num_image + sentence['value']
                sentence['value'] = sentence['value'].strip()
    return sources

def preprocess_multimodal_special(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_SPEECH_TOKEN in sentence['value'] and (DEFAULT_SPEECH_TOKEN + '\n') not in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_SPEECH_TOKEN, (DEFAULT_SPEECH_TOKEN + '\n'))
    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False
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

    if has_speech:
        input_ids = torch.stack([tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    if conv.sep_style == conversation_lib.SeparatorStyle.TWO:

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_speech:
                    round_len = len(tokenizer_speech_token(rou, tokenizer))
                    instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    elif conv.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        # Mask targets
        sep = '<|im_start|>assistant\n'
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            raw_rounds = conversation.split('<|im_end|>\n')
            cur_len = 0
            rounds = []
            now_str = ''
            for rou in raw_rounds:
                if len(rou) > 0:
                    rou = rou + '<|im_end|>\n'
                    if rou.startswith('<|endoftext|>'):
                        rounds[-1] = rounds[-1] + '<|endoftext|>'
                        rou = rou.replace('<|endoftext|>', '')
                        if len(rou.strip()) == 0:
                            continue
                    if '<|im_start|>assistant\n' in rou:
                        now_str += rou
                        rounds.append(now_str)
                        now_str = ''
                    else:
                        now_str += rou

            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_speech:
                    round_len = len(tokenizer_speech_token(rou, tokenizer))
                    instruction_len = len(tokenizer_speech_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                try:
                    is_legacy = tokenizer.legacy
                except:
                    is_legacy = True

                if i != 0 and not is_legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch for QWEN2: {cur_len} vs. {total_len}."
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
        assert DEFAULT_SPEECH_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_SPEECH_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_speech_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_speech_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_speech: bool = False, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    # im_start, im_end = tokenizer.additional_special_tokens_ids

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and has_speech and "<speech><image>" in sentence["value"]:
                if sentence["value"].startswith("<speech><image>"):
                    _input_id = tokenizer(role).input_ids + nl_tokens + [SPEECH_TOKEN_INDEX] + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<speech><image>") :]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = []
                    split_value = sentence["value"].split('<speech><image>\n')
                    _input_id += tokenizer(role).input_ids + nl_tokens
                    for idx, cur_value in enumerate(split_value):
                        if idx == len(split_value) - 1:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [im_end] + nl_tokens
                        else:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [SPEECH_TOKEN_INDEX] + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens
            elif has_image and has_speech and "<speech>" in sentence["value"] and "<image>" in sentence["value"]:
                _input_id = []
                split_value = sentence["value"].split('<image>\n')
                split_value_ = []
                for cur_value in split_value:
                    split_value_.extend(cur_value.split('<speech>\n'))
                _input_id += tokenizer(role).input_ids + nl_tokens
                for idx, cur_value in enumerate(split_value_):
                    if idx == len(split_value_) - 1:   # after <speech>
                        _input_id = _input_id + tokenizer(cur_value).input_ids + [im_end] + nl_tokens
                    elif idx == len(split_value_) - 2:   # after <image>
                        _input_id = _input_id + tokenizer(cur_value).input_ids + [SPEECH_TOKEN_INDEX] + nl_tokens
                    else:
                        _input_id = _input_id + tokenizer(cur_value).input_ids + [IMAGE_TOKEN_INDEX] + nl_tokens
            elif has_speech and "<speech>" in sentence["value"]:
                if sentence["value"].startswith("<speech>"):
                    _input_id = tokenizer(role).input_ids + nl_tokens + [SPEECH_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<speech>") :]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = []
                    split_value = sentence["value"].split('<speech>\n')
                    _input_id += tokenizer(role).input_ids + nl_tokens
                    for idx, cur_value in enumerate(split_value):
                        if idx == len(split_value) - 1:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [im_end] + nl_tokens
                        else:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [SPEECH_TOKEN_INDEX] + nl_tokens
            elif has_image and "<image>" in sentence["value"]:
                _input_id = []
                split_value = sentence["value"].split('<image>\n')
                _input_id += tokenizer(role).input_ids + nl_tokens
                for idx, cur_value in enumerate(split_value):
                    if idx == len(split_value) - 1:
                        if cur_value == '':
                            _input_id = _input_id + [im_end] + nl_tokens
                        else:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [im_end] + nl_tokens
                    else:
                        if cur_value == '':
                            _input_id = _input_id+ [IMAGE_TOKEN_INDEX] + nl_tokens
                        else:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [IMAGE_TOKEN_INDEX] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_speech: bool = False,
    has_image: bool = False,
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
        return preprocess_v1(sources, tokenizer, has_speech=has_speech)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_speech=has_speech, has_image=has_image)
    raise NotImplementedError

def read_audio_patch(patch_info):
    if isinstance(patch_info, str):
        audio_file_name = patch_info
        speechs, samplerate = librosa.load(audio_file_name, sr=16000)
        if len(speechs.shape) > 1:
            speechs = speechs[:, 0]
        return speechs

    audio_file_name = patch_info['patch']
    start_bytes = int(patch_info['start_num'])

    if isinstance(patch_info['size'], int):
        file_size = int(patch_info['size'])
        with open(audio_file_name, 'rb') as f:
            f.seek(start_bytes)
            speechs, samplerate = librosa.load(io.BytesIO(f.read(file_size)), sr=16000)
            if len(speechs.shape) > 1:
                speechs = speechs[:, 0]
    elif isinstance(patch_info['size'], list):
        file_size = patch_info['size']
        speechs = []
        offset = 0
        with open(audio_file_name, 'rb') as f:
            for cur_size in file_size:
                f.seek(start_bytes + offset)
                speech, samplerate = librosa.load(io.BytesIO(f.read(cur_size)), sr=16000)
                if len(speech.shape) > 1:
                    speech = speech[:, 0]
                speechs.append(speech)
                offset += cur_size
    return speechs


def read_image_patch(patch_info):
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
    else:
        image_file_name = patch_info['patch']
        start_bytes = int(patch_info['start_num'])
        file_size = int(patch_info['size'])

        with open(image_file_name, 'rb') as f:
            f.seek(start_bytes)
            if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                image = Image.open(io.BytesIO(base64.b64decode(f.read(file_size).decode()))).convert("RGB")
            else:
                image = Image.open(io.BytesIO(f.read(file_size))).convert("RGB")
    return image

def read_video_patch(patch_info):
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
    else:
        image_file_name = patch_info['patch']
        start_bytes = int(patch_info['start_num'])
        file_size = patch_info['size'] # list of int
        total_file_size = 0
        images_all = []
        with open(image_file_name, 'rb') as f:
            for idx in range(len(file_size)):
                f.seek(start_bytes + total_file_size)
                if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                    image = Image.open(io.BytesIO(base64.b64decode(f.read(int(file_size[idx])).decode()))).convert("RGB")
                else:
                    if 'sharegpt4o' in image_file_name or 'ShareGPT4Video/new_patch' in image_file_name or 'cinepile' in image_file_name or 'nextqa' in image_file_name or 'perceptiontest' in image_file_name:
                        byte_str = io.BytesIO(f.read(int(file_size[idx])))
                        array = np.frombuffer(byte_str.getvalue(), dtype=np.uint8)
                        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                        image = Image.fromarray(image)
                    else:
                        image = Image.open(io.BytesIO(f.read(int(file_size[idx])))).convert("RGB")
                images_all.append(image)
                total_file_size += int(file_size[idx])
    return images_all

def read_video_file(file_path):
    from decord import VideoReader, cpu
    vr = VideoReader(file_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.arange(0, total_frame_num, dtype=int).tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]
    return video

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
        self.mel_size = 128

    def __len__(self):
        return len(self.list_data_dict)

    def process_audio(self, audio_file):
        speech_wav = read_audio_patch(audio_file)
        speech_wav = speech_wav.astype(np.float32)
        CHUNK_LIM = 480000
        speechs = []
        speech_wavs = []

        if len(speech_wav) <= CHUNK_LIM:
            speech = whisper.pad_or_trim(speech_wav)
            speech_wav = whisper.pad_or_trim(speech_wav)
            speechs.append(speech)
            speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
        else:
            for i in range(0, len(speech_wav), CHUNK_LIM):
                chunk = speech_wav[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                speechs.append(chunk)
                speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
        mels = []
        for chunk in speechs:
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=self.mel_size).permute(1, 0).unsqueeze(0)
            mels.append(chunk)

        mels = torch.cat(mels, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)
        if mels.shape[0] > 25:
            mels = mels[:25]
            speech_wavs = speech_wavs[:25]

        speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])

        return mels, speech_length, speech_chunks, speech_wavs

    def process_image(self, image_file):
        if type(image_file) is str:
            image = Image.open(image_file).convert('RGB')
        elif type(image_file) is dict:
            image = read_image_patch(image_file)
        else:
            raise ValueError(f"Unknown image file type: {type(image_file)}, {image_file}")
        image_size = image.size
        image, image_padded = process_anyres_highres_image(image, self.data_args.image_processor)

        return (image, image_padded), image_size, "image"
    
    def process_video(self, video_file):
        if isinstance(video_file, str):
            video = read_video_file(video_file)
        else:
            video = read_video_patch(video_file)
        video_processed = []

        cur_frames_upbound = self.data_args.frames_upbound

        if cur_frames_upbound > 0:
            if len(video) > cur_frames_upbound:
                uniform_sampled_frames = np.linspace(0, len(video) - 1, cur_frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
            else:
                frame_idx = None

        for idx, frame in enumerate(video):
            frame = process_anyres_video(frame, self.data_args.image_processor)
            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))
        
        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
        
        video_processed = torch.cat(video_processed, dim=0)

        video_processed = (video_processed, video_processed)
        return (video_processed, (384, 384), "video"), frame_idx

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300
        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        has_speech = ('audio' in self.list_data_dict[i] or 'audio_q' in self.list_data_dict[i])
        has_image = ('image' in self.list_data_dict[i]) or ('video' in self.list_data_dict[i]) or ('video_long' in self.list_data_dict[i])
        

        if 'video' in sources[0] and 'audio' in sources[0]:  # video + audio
            assert 'audio' in sources[0]
            video_file = self.list_data_dict[i]['video']

            audio_file = self.list_data_dict[i]['audio']
            audio, audio_length, audio_chunks, speech_wav = self.process_audio(audio_file)
            # audio = [audio]
            
            video, _ = self.process_video(video_file)
            video = [video]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

        elif 'audio' in sources[0]:  # audio only
            audio_file = self.list_data_dict[i]['audio']
            audio, audio_length, audio_chunks, speech_wav = self.process_audio(audio_file)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        elif 'audio_q' in sources[0] and 'image' in sources[0]: # audio + image
            audio_file = self.list_data_dict[i]['audio_q']
            audio, audio_length, audio_chunks, speech_wav = self.process_audio(audio_file)
            image_file = self.list_data_dict[i]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            sources[0]['conversations'][0]['value'] = sources[0]['text_q']

            sources = preprocess_multimodal_special(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        elif 'audio_q' in sources[0]: # audio + text
            audio_file = self.list_data_dict[i]['audio_q']
            audio, audio_length, audio_chunks, speech_wav = self.process_audio(audio_file)
            sources[0]['conversations'][0]['value'] = sources[0]['text_q']
            sources = preprocess_multimodal_special(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        elif 'video' in sources[0]: # pure video
            video_file = self.list_data_dict[i]['video']
            video, _ = self.process_video(video_file)
            video = [video]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        elif 'image' in sources[0]: # pure image
            image_file = self.list_data_dict[i]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        else: # pure text
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_speech=has_speech,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # audio exist in the data
        if 'audio' in self.list_data_dict[i] or 'audio_q' in self.list_data_dict[i]:
            data_dict['speech'] = audio
            data_dict['speech_lengths'] = audio_length
            data_dict['speech_chunks'] = audio_chunks
            data_dict['speech_wav'] = speech_wav

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        if 'video' in self.list_data_dict[i]:
            data_dict['image'] = video
        
        if self.data_args.is_multimodal and 'image' not in self.list_data_dict[i] and 'video' not in self.list_data_dict[i]:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = [(
                    (torch.zeros(1, 3, crop_size['height'], crop_size['width']), torch.zeros(1, 3, crop_size['height'], crop_size['width'])),
                    (crop_size['width'], crop_size['height']),
                    "text"
                ),]
        if self.data_args.is_multimodal and 'audio' not in self.list_data_dict[i] and 'audio_q' not in self.list_data_dict[i]:
            data_dict['speech'] = torch.zeros(1, 3000, 128)
            data_dict['speech_lengths'] = torch.LongTensor([3000])
            data_dict['speech_chunks'] = torch.LongTensor([1])
            data_dict['speech_wav'] = torch.zeros([1, 480000])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower() or "oryx" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                raise NotImplementedError
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            speeches_lengths = [instance['speech_lengths'] for instance in instances]
            speeches_chunks = [instance['speech_chunks'] for instance in instances]
            speeches_wav = [instance['speech_wav'] for instance in instances]

            batch['speech_chunks'] = [au for audio_list in speeches_chunks for au in audio_list]
            batch['speech_chunks'] = torch.stack(batch['speech_chunks'])
            
            batch['speech'] = [au for audio_list in speeches for au in audio_list]
            
            batch['speech_lengths'] = [au for audio_list in speeches_lengths for au in audio_list]
            batch['speech_lengths'] = torch.stack(batch['speech_lengths'])

            batch['speech_wav'] = [au for audio_list in speeches_wav for au in audio_list]
            batch['speech_wav'] = torch.stack(batch['speech_wav'])


            if all(x is not None and x.shape == speeches[0][0].shape for x in batch['speech']):
                batch['speech'] = torch.stack(batch['speech'])

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image_sizes'] = [im[1] for im_list in images for im in im_list]
            batch['modalities'] = [im[2] for im_list in images for im in im_list]
            images_lowres = [im[0][0] for im_list in images for im in im_list]
            images_highres = [im[0][1] for im_list in images for im in im_list]
            batch['images_highres'] = images_highres
            if all(x is not None and x.shape == images_lowres[0].shape for x in images_lowres):
                batch['images'] = torch.stack(images_lowres)
            else:
                batch['images'] = images_lowres
                
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


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model = OlaQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        training_args.ddp_find_unused_parameters = False

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=True
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.to(dtype=compute_dtype, device=training_args.device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir, 
        model_max_length=training_args.model_max_length, 
        padding_side="right")

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    tokenizer.add_tokens(
        ['<|ocr_start|>', '<|ocr_end|>', '<|face_start|>', '<|face_end|>', '<|mm_pad|>'],
        special_tokens=True
    )
    print("### Added Special tokens.")

    tokenizer.pad_token = '<|mm_pad|>'

    print(conversation_lib.default_conversation)

    model.get_model().initialize_speech_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    vision_tower.image_processor.do_resize = training_args.do_resize
    vision_tower.image_processor.do_center_crop = training_args.do_center_crop
    
    data_args.image_processor = vision_tower.image_processor

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
    if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
        model.requires_grad_(False)
    if model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    if model_args.tune_mm_vision_resampler:
        for p in model.get_model().vision_resampler.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
    if training_args.freeze_mm_vision_resampler:
        for p in model.get_model().vision_resampler.parameters():
            p.requires_grad = False

    model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
    if training_args.unfreeze_mm_vision_tower:
        vision_tower.requires_grad_(True)
    
    model.config.freeze_mm_vision_tower = training_args.freeze_mm_vision_tower
    if training_args.freeze_mm_vision_tower:
        for p in vision_tower.parameters():
            p.requires_grad = False

    data_args.is_multimodal = True

    model.config.freeze_speech_adapter = training_args.freeze_speech_adapter
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.speech_projector_lr = training_args.speech_projector_lr
    model.config.mm_speech_encoder_lr = training_args.mm_speech_encoder_lr
    model.config.tune_speech_adapter = training_args.tune_speech_adapter = model_args.tune_speech_adapter
    
    speech_encoder = model.get_speech_encoder()
    if speech_encoder is not None:

        speech_encoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        if model_args.tune_speech_adapter:
            model.requires_grad_(False)
            for p in model.get_model().speech_projector.parameters():
                p.requires_grad = True

        if training_args.freeze_speech_adapter:
            for p in model.get_model().speech_projector.parameters():
                p.requires_grad = False

            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

            if hasattr(speech_encoder, "fix_models"):
                speech_encoder.fix_models()
    
        if model_args.fix_speech_encoder:
            speech_encoder.requires_grad_(False)

    if model_args.resume_from is not None:
        resume_path = model_args.resume_from
        if resume_path.endswith('.ckpt') or resume_path.endswith('.pth'):
            sd = torch.load(model_args.resume_from, map_location='cpu')
            rank0_print(f'### loading ckpt from {model_args.resume_from}')
            msg = model.load_state_dict(sd, strict=False)
            rank0_print(msg)
        elif os.path.isdir(resume_path):
            files = glob.glob(os.path.join(resume_path, '*.safetensors'))
            rank0_print(f'### loading from {files}')
            sd = {}
            for file in files:
                sd.update(safetensor_load_file(file))
            msg = model.load_state_dict(sd, strict=False)
            print(msg)
        else:
            raise ValueError(f'### resume_from {model_args.resume_from} not supported')

    total_trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(f'train param: {name}')
            total_trainable_params += p.numel()

    rank0_print(f'#### total trainable params: {total_trainable_params//1000000}M')

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    trainer = OlaTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
