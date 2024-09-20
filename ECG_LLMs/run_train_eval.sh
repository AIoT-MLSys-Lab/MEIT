export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=12 
TOTAL_BATCH_SIZE=24 # 144
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training

    CUDA_VISIBLE_DEVICES=0 python /home/wan.512/ECG_LLMs/ECG_gen/finetune_ecgllm_with_lora.py \
    --model_name_or_path /local/scratch0/data_and_checkpoint/models/llama-7b \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name /local/scratch0/data_and_checkpoint/models/llama-7b \
    --use_slow_tokenizer \
    --train_file /home/wan.512/ECG_LLMs/ECG_gen/instruct_data/mimic_ecg.jsonl \
    --max_seq_length 128 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 1000 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir /home/wan.512/ECG_LLMs/output_lora/ecg_llm_lora \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 10 \
    --use_ecg_llama \

    
    # &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --save_tokenizer
