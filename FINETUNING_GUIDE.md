# Finetuning Guide: Using Your Optimization Data

This guide explains how to convert your collected optimization data and use it for finetuning an LLM.

## Overview

Your original data in `train_data_4o_new_100.json` contains optimization problems with the following structure:
- `task_id`: Problem identifier
- `description`: The problem statement
- `ground_truth`: Expected answer
- `correct_program`: The solution code
- Additional metadata (tags, clusters, etc.)

The finetuning code expects data in one of two formats:
1. **Messages format**: Conversational format with `system`, `user`, and `assistant` roles
2. **Prompt/Completion format**: Simple `prompt` and `completion` fields

## Step 1: Convert Your Data

Use the provided conversion script to transform your data:

### Convert to Messages Format (Recommended)
```bash
python convert_data_for_finetuning.py \
    --input_file train_test_data/train_data_4o_new_100.json \
    --output_file train_test_data/converted_data_messages.json \
    --format messages
```

### Convert to Prompt/Completion Format
```bash
python convert_data_for_finetuning.py \
    --input_file train_test_data/train_data_4o_new_100.json \
    --output_file train_test_data/converted_data_prompt_completion.json \
    --format prompt_completion
```

### Test with a Small Sample
```bash
python convert_data_for_finetuning.py \
    --input_file train_test_data/train_data_4o_new_100.json \
    --output_file train_test_data/test_sample.json \
    --format messages \
    --max_samples 10
```

## Step 2: Prepare Training Configuration

Create a training configuration file. Here are examples for different scenarios:

### Example 1: Small Model with LoRA (Recommended for testing)
```json
{
  "model_name_or_path": "microsoft/DialoGPT-small",
  "train_dataset_name_or_path": "train_test_data/converted_data_messages.json",
  "max_seq_length": 1024,
  "preprocessing_num_workers": 4,
  "overwrite_cache": true,
  "max_train_samples": 50,
  "use_lora": true,
  "lora_rank": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "lora_target_modules": ["c_attn", "c_proj"],
  "output_dir": "./output_small_model",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "num_train_epochs": 3,
  "learning_rate": 5e-4,
  "warmup_steps": 100,
  "logging_steps": 10,
  "save_steps": 500,
  "eval_steps": 500,
  "save_total_limit": 2,
  "remove_unused_columns": false,
  "fp16": true,
  "dataloader_pin_memory": false
}
```

### Example 2: Larger Model (e.g., Llama-2-7b) with LoRA
```json
{
  "model_name_or_path": "meta-llama/Llama-2-7b-hf",
  "train_dataset_name_or_path": "train_test_data/converted_data_messages.json",
  "max_seq_length": 2048,
  "preprocessing_num_workers": 8,
  "overwrite_cache": true,
  "use_lora": true,
  "lora_rank": 32,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "output_dir": "./output_llama2_7b",
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "num_train_epochs": 2,
  "learning_rate": 2e-4,
  "warmup_steps": 200,
  "logging_steps": 10,
  "save_steps": 1000,
  "eval_steps": 1000,
  "save_total_limit": 3,
  "remove_unused_columns": false,
  "fp16": true,
  "dataloader_pin_memory": false,
  "use_auth_token": true
}
```

## Step 3: Run Finetuning

### Using Configuration File
```bash
python train/finetune.py config.json
```

### Using Command Line Arguments
```bash
python train/finetune.py \
    --model_name_or_path microsoft/DialoGPT-small \
    --train_dataset_name_or_path train_test_data/converted_data_messages.json \
    --max_seq_length 1024 \
    --preprocessing_num_workers 4 \
    --overwrite_cache \
    --max_train_samples 50 \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[c_attn,c_proj]" \
    --output_dir ./output_small_model \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --remove_unused_columns false \
    --fp16 \
    --dataloader_pin_memory false
```

## Step 4: Monitor Training

The training process will:
1. Load and preprocess your data
2. Apply LoRA (if enabled) to the model
3. Train the model on your optimization problems
4. Save checkpoints and logs

Monitor the training logs for:
- Loss values (should decrease over time)
- Learning rate schedule
- Memory usage
- Training speed

## Step 5: Test Your Finetuned Model

After training, you can test your model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
model_name = "microsoft/DialoGPT-small"  # or your base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./output_small_model")

# Test with a sample problem
test_prompt = """<|system|>
You are an expert in mathematical optimization and operations research. You help solve complex optimization problems by providing clear, correct Python code using optimization libraries like Gurobi, OR-Tools, or SciPy.
<|user|>
Please solve this optimization problem:

A company wants to minimize costs by choosing between two suppliers. Supplier A charges $10 per unit with a fixed cost of $100. Supplier B charges $8 per unit with a fixed cost of $200. The company needs at least 50 units. Which supplier should they choose?
<|assistant|>"""

inputs = tokenizer(test_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Data Format Details

### Messages Format Structure
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in mathematical optimization..."
    },
    {
      "role": "user", 
      "content": "Please solve this optimization problem:\n\n[problem description]"
    },
    {
      "role": "assistant",
      "content": "Here's the solution:\n\n```python\n[code]\n```\n\nThe optimal value is: [answer]"
    }
  ],
  "task_id": "E077",
  "ground_truth": 587428.0,
  "problem_domain": "Facility Location",
  "modeling_type": "MILP",
  "industry_sector": "Logistics"
}
```

### Prompt/Completion Format Structure
```json
{
  "prompt": "Please solve this optimization problem:\n\n[problem description]",
  "completion": "Here's the solution:\n\n```python\n[code]\n```\n\nThe optimal value is: [answer]",
  "task_id": "E077",
  "ground_truth": 587428.0,
  "problem_domain": "Facility Location",
  "modeling_type": "MILP", 
  "industry_sector": "Logistics"
}
```

## Tips and Best Practices

1. **Start Small**: Begin with a small model and subset of data to test the pipeline
2. **Use LoRA**: LoRA is more memory-efficient and often performs better than full finetuning
3. **Monitor Memory**: Large models require significant GPU memory
4. **Data Quality**: Ensure your converted data has proper formatting
5. **Hyperparameters**: Adjust learning rate, batch size, and epochs based on your data size
6. **Validation**: Test your model on unseen optimization problems

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size, use gradient accumulation, or use a smaller model
2. **Data Loading Errors**: Check that your converted JSON file is valid
3. **Tokenization Issues**: Ensure your tokenizer supports the special tokens used in messages format
4. **Training Instability**: Lower learning rate, increase warmup steps, or reduce batch size

### Getting Help:
- Check the training logs for specific error messages
- Verify your data format matches the expected structure
- Ensure all required dependencies are installed
- Test with a small sample first

## Next Steps

After successful finetuning:
1. Evaluate your model on test optimization problems
2. Compare performance with the base model
3. Fine-tune hyperparameters if needed
4. Consider expanding your training dataset
5. Deploy your model for inference
