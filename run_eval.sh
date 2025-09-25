#!/bin/bash

datasets=(
   "complexor.json"
#    "industryor_test.json"
#    "mamo_complex_test.json"
   "mamo_easy.json"
#    "nl4opt_test.json"
#    "nlp4lp_test.json"
   "optibench.json"
   “optmath_bench.json”
#    "logior_test.json"
     "logior.json"
)

for dataset in "${datasets[@]}"; do
    # Using Qwen-7B model
    # CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path Qwen/Qwen2-7B-Instruct --tensor_parallel_size 1 --gpu_memory_utilization 0.9 --test_file ../baseline_test_data/$dataset --output_file results/Qwen2-7B-Instruct/$dataset --verbose
    # CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file ../baseline_test_data/$dataset --output_file results/Meta-Llama-3-8B-Instruct/$dataset --verbose
        CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/models--AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_100 --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file ../baseline_test_data/$dataset --output_file results/AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_100/$dataset --verbose

done

