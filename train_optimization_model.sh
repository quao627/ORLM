#!/bin/bash

# Training script for optimization model using your converted data
# This script uses DeepSpeed ZeRO Stage 3 for efficient distributed training

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR SETUP
# =============================================================================

# Model and Data Paths
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"  # LLaMA-3-8B-Instruct model
# DATA_PATH="train_test_data/converted_data_messages_new_100.json"  # Your converted data
DATA_PATH="train_test_data/converted_data_prompt_completion_new_100.json"  # Your converted data
## use the base model name and add the new model name
SAVE_PATH="/orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/models--AlphaOpt_ORLM_Meta-Llama-3-8B-Instruct_ft"

# Training Configuration
NUM_GPUS=2  # Number of H200 GPUs available
BATCH_SIZE_PER_GPU=4  # Batch size per GPU (optimized for H200 140GB)
TOTAL_BATCH_SIZE=16  # Total effective batch size across all GPUs
PREPROCESSING_NUM_WORKERS=0  # Number of workers for data preprocessing
MAX_SEQ_LENGTH=8192  # Maximum sequence length (can use full context)
LEARNING_RATE=2e-5  # Learning rate (can be higher with larger batch size)
NUM_TRAIN_EPOCHS=5  # Number of training epochs (can train longer)

# Advanced Options
USE_LORA=false  # Set to false for full parameter tuning
USE_AUTH_TOKEN=true  # Required for LLaMA-3 models

# =============================================================================
# CALCULATED PARAMETERS (DO NOT MODIFY)
# =============================================================================

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# =============================================================================
# VALIDATION
# =============================================================================

echo "🚀 Starting LLaMA-3-8B full parameter tuning..."
echo "==============================================="
echo "Model: $MODEL_NAME_OR_PATH"
echo "Data: $DATA_PATH"
echo "Output: $SAVE_PATH"
echo "GPUs: $NUM_GPUS"
echo "Batch size per GPU: $BATCH_SIZE_PER_GPU"
echo "Total batch size: $TOTAL_BATCH_SIZE"
echo "Gradient accumulation steps: $GRADIENT_ACC_STEPS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Learning rate: $LEARNING_RATE"
echo "Training epochs: $NUM_TRAIN_EPOCHS"
echo "Use LoRA: $USE_LORA (FULL PARAMETER TUNING)"
echo "Use Auth Token: $USE_AUTH_TOKEN"
echo "==============================================="

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found at $DATA_PATH"
    echo "Please run the data conversion script first:"
    echo "python convert_data_for_finetuning.py --input_file train_test_data/train_data_4o_new_100.json --output_file $DATA_PATH --format prompt_completion"
    exit 1
fi

# Check if DeepSpeed config exists
if [ ! -f "train/configs/h200_optimized_bf16.json" ]; then
    echo "❌ Error: DeepSpeed config not found at train/configs/h200_optimized_bf16.json"
    exit 1
fi

echo "✅ All files found. Starting training..."

# =============================================================================
# TRAINING COMMAND
# =============================================================================

torchrun \
    --nproc_per_node $NUM_GPUS \
    -m train.finetune \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed train/configs/h200_optimized_bf16.json \
    --overwrite_output_dir \
    --bf16 True \
    --use_lora $USE_LORA \
    --use_auth_token $USE_AUTH_TOKEN \
    --remove_unused_columns False

echo "🎉 Training completed!"
echo "Model saved to: $SAVE_PATH"
