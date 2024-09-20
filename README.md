# MEIT
## Easy steps for efficient implementations

#### Step 1: download data and preprocess data

- 1.download data from google drive to your linux device:
Google drive link: #############. 

- 2.proprocess data: get into the 'config.yaml' file to set up the link of downloaded data:

```
if mimic:
dataset:
  dataset_name: 'mimic'
  ## this is for mimic dataset 21k
  ecg_path: 'xxxx' # add your image file path here
  text_path: 'xxx_train.csv'

if ptbxl:
dataset:
  dataset_name: 'ptbxl'
  ## this is for PTB-XL dataset 21k
  ecg_path: '/fs/scratch/PAS2473/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/' # add your image file path here
  text_path: '/users/PAS2473/brucewan666/ECG/ECG/instruct_data/RG_en_ptbxl_train.csv'
```

- 3.run preprocess data: 

```
get into preprocess_ecg.py 
set the path of yours (an example of ptbxl): 

build_instruct_dataset(ecg_name='ptbxl',save_path='/users/PAS2473/brucewan666/ECG/ECG/instruct_data/ptbxl_ecg_train.jsonl') # mimic

```

- 4.set up environment：

```
pip install -r requirements.txt
```

- 5.run ecg_instruction_tuning data and inference with only one ECG_instruction, give an example of mimic-ecg data:

```
export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=64 # 144 50277
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
    # --use_deepspeed \
    # --deepspeed_config_file /home/wan.512/ECG_LLMs/open-instruct/ds_configs/stage3_no_offloading_accelerate.conf \
# Lora training
accelerate launch --main_process_port 31225 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    /users/PAS2473/brucewan666/ECG/ECG/finetune_ecgllm_with_lora_mimic.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
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
    --output_dir /fs/scratch/PAS2473/zhongwei_save_ckpt/gpt2_large_lora_ckpt \
    --with_tracking \
    --report_to tensorboard \
    --use_ecg_llm \
    --dev_ratio 0.1 \
    --val_test_ratio 0.1 \
    --logging_steps 100 \
    --eval_step 3200 \
    --test_step 4000 \
    --llm_type llama_2 \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models

```