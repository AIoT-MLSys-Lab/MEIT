#!/bin/bash

#SBATCH --job-name=mistral_015       # 作业名称
#SBATCH --account=PAS2473		    # Project ID
#SBATCH --output=/users/PAS2473/brucewan666/ECG/ECG/output_logs/mistral_015.log        # 输出日志文件
#SBATCH --error=/users/PAS2473/brucewan666/ECG/ECG/output_logs/mistral_015_error.log         # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=1        # GPU per node
#SBATCH --mem=80G                   # 内存限制
#SBATCH --time=50:00:00             # 作业运行时间限制

# 运行命令或脚本 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
source $HOME/anaconda3/bin/activate /users/PAS2473/brucewan666/anaconda3/envs/flashattn
# module load cuda 


export CUDA_VISIBLE_DEVICES=3

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=64 # 144 50277
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
    # --use_deepspeed \
    # --deepspeed_config_file /home/wan.512/ECG_LLMs/open-instruct/ds_configs/stage3_no_offloading_accelerate.conf \
# Lora training
accelerate launch --main_process_port 31226 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    /users/PAS2473/brucewan666/ECG/ECG/finetune_ecgllm_with_lora_mimic_robutness.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --tokenizer_name mistralai/Mistral-7B-v0.1 \
    --use_slow_tokenizer \
    --train_file /users/PAS2473/brucewan666/ECG/ECG/instruct_data/mimic_ecg.jsonl \
    --max_seq_length 128 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir /fs/scratch/PAS2473/robutness_test/mistral_015 \
    --with_tracking \
    --report_to tensorboard \
    --use_ecg_llm \
    --dev_ratio 0.01 \
    --val_test_ratio 0.1 \
    --logging_steps 100 \
    --eval_step 3200 \
    --test_step 4000 \
    --llm_type mistral_v1 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --max_proportion 0.15 


