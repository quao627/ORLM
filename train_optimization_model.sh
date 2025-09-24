#!/bin/bash

# Training script for optimization model using your converted data
# This script uses DeepSpeed ZeRO Stage 3 for efficient distributed training

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR YOUR SETUP
# =============================================================================

# Model and Data Paths
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"  # Change to your base model
DATA_PATH="train_test_data/converted_data_messages.json"  # Your converted data
SAVE_PATH="./output_optimization_model"  # Where to save the model

# Training Configuration
NUM_GPUS=4  # Number of GPUs available
BATCH_SIZE_PER_GPU=1  # Batch size per GPU (adjust based on GPU memory)
TOTAL_BATCH_SIZE=16  # Total effective batch size across all GPUs
PREPROCESSING_NUM_WORKERS=8  # Number of workers for data preprocessing
MAX_SEQ_LENGTH=4096  # Maximum sequence length (adjust based on your data)
LEARNING_RATE=2e-5  # Learning rate (typical for LLM finetuning)
NUM_TRAIN_EPOCHS=3  # Number of training epochs

# Advanced Options
USE_LORA=true  # Set to true to use LoRA (recommended for efficiency)
LORA_RANK=32  # LoRA rank
LORA_ALPHA=32  # LoRA alpha
LORA_DROPOUT=0.1  # LoRA dropout
LORA_TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"  # LoRA target modules

# =============================================================================
# CALCULATED PARAMETERS (DO NOT MODIFY)
# =============================================================================

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# =============================================================================
# VALIDATION
# =============================================================================

echo "🚀 Starting optimization model training..."
echo "=========================================="
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
echo "Use LoRA: $USE_LORA"
echo "=========================================="

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found at $DATA_PATH"
    echo "Please run the data conversion script first:"
    echo "python convert_data_for_finetuning.py --input_file train_test_data/train_data_4o_new_100.json --output_file $DATA_PATH --format messages"
    exit 1
fi

# Check if DeepSpeed config exists
if [ ! -f "train/configs/stage3_no_offloading_bf16.json" ]; then
    echo "❌ Error: DeepSpeed config not found at train/configs/stage3_no_offloading_bf16.json"
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
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed train/configs/stage3_no_offloading_bf16.json \
    --overwrite_output_dir \
    --bf16 True \
    --use_lora $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --remove_unused_columns False

echo "🎉 Training completed!"
echo "Model saved to: $SAVE_PATH"
