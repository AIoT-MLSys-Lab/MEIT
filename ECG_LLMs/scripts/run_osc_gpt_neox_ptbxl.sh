#!/bin/bash

#SBATCH --job-name=gpt_neox_2       # 作业名称
#SBATCH --account=PAS2473		    # Project ID
#SBATCH --output=/users/PAS2473/brucewan666/ECG/ECG/output_logs/gpt_neox_ecg.log        # 输出日志文件
#SBATCH --error=/users/PAS2473/brucewan666/ECG/ECG/output_logs/gpt_neox_ecg_error.log         # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=2        # GPU per node
#SBATCH --mem=128G                   # 内存限制
#SBATCH --time=70:00:00             # 作业运行时间限制


export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=64 # 144 50277
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
    # --use_deepspeed \
    # --deepspeed_config_file /home/wan.512/ECG_LLMs/open-instruct/ds_configs/stage3_no_offloading_accelerate.conf \
# Lora training
accelerate launch --main_process_port 31254 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /users/PAS2473/brucewan666/ECG/ECG/ds_configs/stage3_no_offloading_accelerate.conf \
    /users/PAS2473/brucewan666/ECG/ECG/semanic_test.py \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --tokenizer_name EleutherAI/gpt-neox-20b \
    --use_slow_tokenizer \
    --train_data_path  /users/PAS2473/brucewan666/ECG/ECG/instruct_data/ptbxl_ecg_train.jsonl \
    --test_data_path /users/PAS2473/brucewan666/ECG/ECG/instruct_data/ptbxl_ecg_test.jsonl \
    --val_data_path /users/PAS2473/brucewan666/ECG/ECG/instruct_data/ptbxl_ecg_val.jsonl \
    --max_seq_length 128 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.04 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir /fs/scratch/PAS2473/semantic_test/gpt_neox_lora_ckpt_ptbxl_2 \
    --with_tracking \
    --report_to tensorboard \
    --use_ecg_llm \
    --dev_ratio 0.1 \
    --val_test_ratio 0.1 \
    --logging_steps 50 \
    --eval_step 400 \
    --test_step 200 \
    --llm_type gpt_neox \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models


# /ecg_llama_7b_new_2   ecg_llama_7b_new_2_llama2_chat_hf  ecg_llama_7b_new_2_Llama-2-7b-hf metric_error_test
# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --save_tokenizer


# mistralai/Mistral-7B-v0.1     mistralai/Mistral-7B-Instruct-v0.2
# meta-llama/Llama-2-7b-hf
# facebook/opt-6.7b
# gpt2-large
# bigscience/bloom-7b1

# llama2 / bloom / opt / mistral conda activate ECG_LLMs