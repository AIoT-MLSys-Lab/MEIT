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
from ECG_gen.ecg_llm_for_llama import ECG_LlamaForCausalLM
import wfdb
import numpy as np
from pathlib import Path
from ECG_gen.ecg_llm_for_llama import ECG_LlamaForCausalLM
from ECG_gen.finetune_ecg_llama_with_lora import encode_with_prompt_completion_format, encode_with_messages_format
from utils_ecg import get_rouge_n_gram
from eval_metrics import compute_metric_scores

logger = get_logger(__name__)



def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    logger.info("Loading the ECG_LLM model...")

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
                                      )

                                    
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    logger.info("Loading the ECG_LLM model success ######################")

    ############################## start to preprocess test data ###############

    data_files = {}
    dataset_args = {}
    if args.dataset_name is not None: # split the train and valid ECG data here 
        data_files["train"] = args.dataset_name
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    ######################### split the train and test dataset here ##################################

    if args.dev_ratio > 1e-6:
        train_test_split_dataset = raw_datasets['train'].train_test_split(args.dev_ratio)

    
    
    # train_dataset = train_test_split_dataset['train']
    # test_dataset = train_test_split_dataset['test'] # 

    # Preprocessing the datasets.
    if "prompt" in train_test_split_dataset['test'].column_names and "completion" in train_test_split_dataset['test'].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in train_test_split_dataset['test'].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    # with accelerator.main_process_first():  ######### deal with the labels and ecg data #############
    lm_datasets = train_test_split_dataset.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in train_test_split_dataset["test"].column_names if name not in ["input_ids", "labels", "attention_mask", "ecg"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    # if args.dev_ratio > 1e-6:
    #     split_datasets = lm_datasets["train"].train_test_split(test_size=args.dev_ratio)
    
    # test_dataset = split_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(lm_datasets)), 3):
        logger.info(f"Sample {index} of the training set: {lm_datasets[index]}.")

    test_dataset = lm_datasets['test']
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.llama_model, padding="longest"), ########## modified here ###############
        batch_size=args.per_device_train_batch_size
    )

    ###########################################################################################

    logger.info("Start to eval ECG_LLM model test data ######################")

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

    with torch.no_grad():

        for num_idx, batch in tqdm(enumerate(test_dataloader)):

            iter_batch += 1

            if model.device.type == "cuda":

                input_ids = batch['input_ids'].cuda()
                reference_ids = input_ids.clone()
                labels = batch['labels'].cuda()
                ecg = batch['ecg'].cuda()

                mask = labels.eq(-100)
                input_ids_for_inference = input_ids.masked_select(mask).view(input_ids.size(0), -1).cuda()

                output_ids = model.generate(
                    input_ids=input_ids_for_inference,
                    ecg = ecg,
                )
                # check here:
                generate_reports = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                reference_reports = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)


                # bleu_1, bleu_2, bleu_3, bleu_4, rouge_1, rouge_2, rouge_l, meteor, cider
                nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
                language_model_scores = compute_metric_scores(nlg_metrics, generate_reports, reference_reports, train_test_split_dataset['test'])
                eval_bleu_1 += language_model_scores['bleu_1']
                eval_bleu_2 += language_model_scores['bleu_2']
                eval_bleu_3 += language_model_scores['bleu_3']
                eval_bleu_4 += language_model_scores['bleu_4']

                eval_rouge_1 += language_model_scores['rouge_1']
                eval_rouge_2 += language_model_scores['rouge_2']
                eval_rouge_l += language_model_scores['rouge_l']

                eval_meteor += language_model_scores['meteor']
                eval_cider += language_model_scores['cider']
    
    avg_bleu_1 = eval_bleu_1 / iter_batch
    avg_bleu_2 = eval_bleu_2 / iter_batch
    avg_bleu_3 = eval_bleu_3 / iter_batch
    avg_bleu_4 = eval_bleu_4 / iter_batch

    avg_rouge_1 = eval_rouge_1 / iter_batch
    avg_rouge_2 = eval_rouge_2 / iter_batch
    avg_rouge_l = eval_rouge_l / iter_batch

    avg_meteor = eval_meteor / iter_batch
    avg_cider = eval_cider / iter_batch

    logger.info(f"  avg_bleu_1: {avg_bleu_1}, avg_bleu_2: {avg_bleu_2}, avg_bleu_3: {avg_bleu_3}, avg_bleu_4: {avg_bleu_4}")
    logger.info(f"  avg_rouge_1: {avg_rouge_1}, avg_rouge_2: {avg_rouge_2}, avg_rouge_l: {avg_rouge_l}")
    logger.info(f"  avg_meteor: {avg_meteor}, avg_cider: {avg_cider}")

    #### add evaluation here #################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='/home/wan.512/ECG_LLMs/ECG_gen/instruct_data/mimic_ecg.jsonl',
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )

    parser.add_argument("--lora_model_name_or_path", 
                        type=str, 
                        required=True)
    
    parser.add_argument("--base_model_name_or_path", 
                        type=str, 
                        required=False)

    parser.add_argument("--save_tokenizer", 
                        action="store_true")

    parser.add_argument("--use_fast_tokenizer", 
                        action="store_true")

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
        "--ecg_layer_idxs",
        type=str,
        default="all",
        help=(
            "31,30,29,28,27"
        ),
    )

    args = parser.parse_args()

    main(args)