#!/usr/bin/env python
# coding=utf-8
from typing import Dict, List, Optional
import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from modeling_ecg_llama import LlamaForCausalLM, LlamaModel
import transformers
from torchvision.transforms import transforms
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    # DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    LlamaConfig
)
from ecg_data_collator import DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from ecg_llm_for_llama import ECG_LlamaForCausalLM
import wfdb
import numpy as np
from pathlib import Path
from eval_metrics import compute_metric_scores
from transformers import BloomForCausalLM, MistralForCausalLM, GPT2LMHeadModel, GPTNeoForCausalLM, GPTJForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM
logger = get_logger(__name__)

def get_state_dict(state_dict): # get state dict containing trainable parameters
    state_dict = state_dict
    filtered_state_dict = {}

    name_list = []

    for k, v in state_dict.items():
        if 'ecg_model' in k:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()
            name_list.append(k)

    print('########### filtered_state_dict #########', name_list)

    return filtered_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    
    ######################## new add #############################################################

    parser.add_argument(
        "--use_ecg_llama",
        action="store_true",
        help=(
            "Use ecg_llama and lora to finetuning LLMs using multimodal dataset."
        ),
    )

    parser.add_argument(
        "--ecg_model_type",
        type=str,
        default="ResNet50",
        help=(
            'ResNet18, ResNet34, ResNet50, ResNet101, ResNet152'
        ),
    )

    parser.add_argument(
        "--ecg_layer_idxs",
        type=str,
        default="all",
        help=(
            "31,30,29,28,27"
        ),
    )


    parser.add_argument(
        "--ecg_data_name",
        type=str,
        default="mimic",
        help=(
            "mimic or ptb "
        ),
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=100,
        help="Total number of eval_step steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--test_step",
        type=int,
        default=200,
        help="Total number of eval_step steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the development set, should be between 0.0 and 1.0.",
    )

    

    parser.add_argument(
        "--val_test_ratio",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the development set, should be between 0.0 and 1.0.",
    )

    parser.add_argument(
        "--train_or_test",
        type=str,
        default="train",
        help=(
            "train or test the ECG_LLMs model "
        ),
    )

    parser.add_argument(
        "--ecg_model_ckpt_path",
        type=str,
        default=None,
        help=(
            "checkpoint of ecg_model  "
        ),
    )

    parser.add_argument(
        "--llm_type",
        type=str,
        default=None,
        help=(
            "choose the LLM backbone to train model  "
        ),
    )



    parser.add_argument("--lora_model_name_or_path", 
                        type=str)




    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']


    dataset_path = example['ecg_path']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                ########################## add prompt for ecg signal here ################################
                prompt = "Given the ECG signal embeddings, please help me generate an accurate description for this ECG signal embeddings: "

                # message_text += "<|user|>\n" + prompt.strip() + "\n"

                message_text += ""
                
            elif message["role"] == "assistant":
                # message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                message_text += message["content"].strip()
            elif message["role"] == "ecg_path":
                pass
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100 # ignore the loss computation ##################
            
            if message_end_idx >= max_seq_length:
                break


    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'ecg': dataset_path,
    }

def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    llama_name_list = []
    unwrapped_model = accelerator.unwrap_model(model.llama_model)
    unwrapped_ecg_model =  accelerator.unwrap_model(model.ecg_model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    
    # for name, params in state_dict.items():
    #     llama_name_list.append(name)
    # print('The fucking state dict: ', llama_name_list)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            ecg_model_path = os.path.join(output_dir, 'ecg_model.pth')

            ecg_state_dict = get_state_dict(state_dict)
            torch.save(ecg_state_dict, Path(ecg_model_path))
            print('####### save ecg model ok ############')
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )

def save_with_accelerate_test(accelerator, model, tokenizer, output_dir, args):

    ################### Remember to store the finetuned checkpoint of ecg encoder ###################

    # unwrapped_model = accelerator.unwrap_model(model.module.llama_model)
    # # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # # Otherwise, sometimes the model will be saved with only part of the parameters.
    # # Also, accelerator needs to use the wrapped model to get the state_dict.
    # state_dict = accelerator.get_state_dict(model.module.llama_model)

    # unwrapped_model_ecg = accelerator.unwrap_model(model.module.ecg_encoder)

    # state_dict_ecg = accelerator.get_state_dict(model.module.ecg_encoder)
    logger.info(f"Saving model checkpoint to #############  {output_dir}")
    model = accelerator.unwrap_model(model)
    llama_model = model.llama_model

    if args.use_lora:

        if hasattr(llama_model, "pretrained_model"): # for models with valuehead
            backbone_model = getattr(llama_model, "pretrained_model")
        else:
            backbone_model = llama_model
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            # unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            # unwrapped_model_ecg.save_pretrained(output_dir, state_dict=state_dict_ecg)

            # model_to_save = model.module if hasattr(model, "module") else model
            # torch.save(model_to_save.llama_model.state_dict(), output_dir)
            if hasattr(backbone_model, "peft_config"): # peft methods
                backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model)) # save lora weights

            print('save lora checkpoint successfully! ################')
            
    else:
        unwrapped_model = accelerator.unwrap_model(model.module.llama_model)
        unwrapped_model_ecg = accelerator.unwrap_model(model.module.ecg_encoder)
        state_dict = accelerator.get_state_dict(model.module.llama_model)
        state_dict_ecg = accelerator.get_state_dict(model.module.ecg_encoder)

        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )

        unwrapped_model_ecg.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict_ecg
        )


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None: # split the train and valid ECG data here 
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
    # split train val test data here ######################################

    if args.dev_ratio > 1e-6:
        train_test_split_dataset = raw_datasets['train'].train_test_split(args.val_test_ratio)




    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.use_ecg_llama:

        if args.ecg_layer_idxs == 'all':
            ecg_layer_idxs = [idx for idx in range(config.num_hidden_layers)]
        else:
            ecg_layer_idxs = [str(idx) for idx in args.ecg_layer_idxs.split(",")]

        print('############### the ecg layers include ################: ', ecg_layer_idxs)

        model = ECG_LlamaForCausalLM(args=args,
                                      model_name_or_path=args.model_name_or_path, 
                                      config=config, 
                                      tokenizer=tokenizer,
                                      ecg_layer_idxs=ecg_layer_idxs
                                      ) # add ecg_llm here

        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })                           

    else:

        if args.model_name_or_path:
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                device_index = accelerator.local_process_index
                device_map = {"": device_index} # force data-parallel training.
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    load_in_4bit=True,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)


        # no default pad token for llama!
        # here we add all special tokens again, because the default ones are not in the special_tokens_map
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
        elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
            num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.


    if args.use_ecg_llama is False:

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        if args.use_lora:
            if args.use_qlora:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

            logger.info("Initializing LORA model...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=args.lora_rank, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    else:
        print('############ We use fancy ECG_LlamaForCausalLM ################')
         # use ecg_llm here

    # Preprocessing the train and test  datasets. #################################################################################
                   ##################### modified test here for bug #################
    if "prompt" in train_test_split_dataset["train"].column_names and "completion" in train_test_split_dataset["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in train_test_split_dataset["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

        
    
    ######### deal with the labels and ecg train data #############
    with accelerator.main_process_first():  
        train_lm_datasets = train_test_split_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,                  ##################### modified test here for bug #################
            load_from_cache_file=not args.overwrite_cache,         
            remove_columns=[name for name in train_test_split_dataset["train"].column_names if name not in ["input_ids", "labels", "attention_mask", "ecg"]],
            desc="Tokenizing and reformatting instruction data",
        )
        train_lm_datasets.set_format(type="pt")
        train_lm_datasets = train_lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    
    ######### deal with the labels and ecg test data #############

    with accelerator.main_process_first():  
        test_lm_datasets = train_test_split_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in train_test_split_dataset["test"].column_names if name not in ["input_ids", "labels", "attention_mask", "ecg"]],
            desc="Tokenizing and reformatting instruction data",
        )
        test_lm_datasets.set_format(type="pt")
        test_lm_datasets = test_lm_datasets.filter(lambda example: (example['labels'] != -100).any())


    if args.dev_ratio > 1e-6:
        split_datasets = test_lm_datasets["test"].train_test_split(test_size=args.dev_ratio)
    

    train_dataset = train_lm_datasets["train"]
    test_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation: #######################################################
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.llama_model, padding="longest"), ########## modified here ###############
        batch_size=args.per_device_train_batch_size
    )

    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.llama_model, padding="longest"), ########## modified here ###############
        batch_size=32
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.llama_model, padding="longest"), ########## modified here ###############
        batch_size=args.per_device_train_batch_size
    )

    #  ###############################################################################

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)                
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()   

            ############# start to evaluate the validation set of ecg data ##################  
            
            if step > 9 and  step % args.eval_step == 0:
  
                eval_step = 0
                total_eval_loss = 0
                model.eval()
                for eval_step, inputs in enumerate(tqdm(eval_dataloader)):

                    eval_step += 1
                    with torch.no_grad():
                        outputs = model(**inputs, use_cache=False)  
                        eval_loss = outputs.loss
                        # eval_loss = eval_loss.reduce_mean().detach().float().cpu()
                        eval_loss = accelerator.gather(eval_loss).mean().detach().float().cpu().item()
                        total_eval_loss += eval_loss

                eval_avg_loss = total_eval_loss / eval_step

                if args.with_tracking:
                    accelerator.log(
                            {
                                "eval_loss": eval_avg_loss,
                            },
                            step=completed_steps,
                        )

                    logger.info(f"  Eval Step: {step}, Eval Loss: {eval_avg_loss}")

                model.train()

            # Checks if the accelerator has performed an optimization step behind the scenes

            if step > 9 and  step % args.test_step == 0:
                logger.info('Start to test the ECG_model with metic # Bleu 1-4, Meteor, Rouge-1 2, Rouge-L, Cider-D ####################')
                
                model.eval()
                
                iter_batch = 0
                eval_bleu_1 = 0
                eval_bleu_2 = 0
                eval_bleu_3 = 0
                eval_bleu_4 = 0

                eval_rouge_1 = 0
                eval_rouge_2 = 0
                eval_rouge_l = 0

                eval_meteor = 0
                eval_cider = 0
                if accelerator.is_main_process:
                    with torch.no_grad():
                        for num_idx, batch in enumerate(tqdm(eval_dataloader)):

                            input_ids = batch['input_ids']
                            reference_ids = input_ids.clone()
                            labels = batch['labels']
                            ecg = batch['ecg']

                            first_non_minus_100 = (labels != -100).long().argmax(dim=1)
                            input_ids_num = input_ids.size(0)
                            input_ids_for_inference = torch.stack([input_ids[i, :first_non_minus_100[i]] for i in range(input_ids_num)]).to(input_ids.device)

                            output_ids = model.generate(
                                        input_ids=input_ids_for_inference,
                                        ecg = ecg,
                                    )
                            # check here:

                            
                            generate_reports = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                            reference_reports = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)

                            if args.llm_type == 'opt':


                                generate_reports = [ sample.replace('<pad>', '').strip() for sample in generate_reports]

                            else:

                                generate_reports =generate_reports


                            if num_idx == 5:

                                print('generate_reports sample 1', generate_reports[0])
                                print('reference_reports sample 1', reference_reports[0])

                                print('generate_reports sample 2', generate_reports[1])
                                print('reference_reports sample 2', reference_reports[1])

                            # bleu_1, bleu_2, bleu_3, bleu_4, rouge_1, rouge_2, rouge_l, meteor, cider, bert_score
                            nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
                            language_model_scores = compute_metric_scores(nlg_metrics, generate_reports, reference_reports, train_test_split_dataset['test'])
                            eval_bleu_1 += language_model_scores['bleu_1']
                            eval_bleu_2 += language_model_scores['bleu_2']
                            eval_bleu_3 += language_model_scores['bleu_3']
                            eval_bleu_4 += language_model_scores['bleu_4']

                            eval_rouge_1 += language_model_scores['rouge_1']
                            eval_rouge_2 += language_model_scores['rouge_2']
                            eval_rouge_l += language_model_scores['rouge']

                            eval_meteor += language_model_scores['meteor']
                            eval_cider += language_model_scores['cider']

                            iter_batch += 1

                    avg_bleu_1 = eval_bleu_1 / iter_batch
                    avg_bleu_2 = eval_bleu_2 / iter_batch
                    avg_bleu_3 = eval_bleu_3 / iter_batch
                    avg_bleu_4 = eval_bleu_4 / iter_batch

                    avg_rouge_1 = eval_rouge_1 / iter_batch
                    avg_rouge_2 = eval_rouge_2 / iter_batch
                    avg_rouge_l = eval_rouge_l / iter_batch

                    avg_meteor = eval_meteor / iter_batch
                    avg_cider = eval_cider / iter_batch


                    if args.with_tracking:
                        accelerator.log(
                            {
                                "avg_bleu_1": avg_bleu_1,
                                 "avg_bleu_2": avg_bleu_2,
                                  "avg_bleu_3": avg_bleu_3,
                                   "avg_bleu_4": avg_bleu_4,
                                    "avg_rouge_1": avg_rouge_1,
                                     "avg_rouge_2": avg_rouge_2,
                                      "avg_rouge_l": avg_rouge_l,
                                       "avg_meteor": avg_meteor,
                                       "avg_cider": avg_cider,

                            },
                            step=completed_steps,
                        )

                    logger.info(f"  avg_bleu_1: {avg_bleu_1}, avg_bleu_2: {avg_bleu_2}, avg_bleu_3: {avg_bleu_3}, avg_bleu_4: {avg_bleu_4}")
                    logger.info(f"  avg_rouge_1: {avg_rouge_1}, avg_rouge_2: {avg_rouge_2}, avg_rouge_l: {avg_rouge_l}")
                    logger.info(f"  avg_meteor: {avg_meteor}, avg_cider: {avg_cider}")

                model.train()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                            
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)


    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)



if __name__ == "__main__":
    main()
