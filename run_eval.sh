#!/bin/bash

datasets=(
#    "complexor.json"
#    "industryor_test.json"
#    "mamo_complex_test.json"
#    "mamo_easy.json"
#    "nl4opt_test.json"
#    "nlp4lp_test.json"
#    "optibench.json"
#    "logior_test.json"
     "logior.json"
)

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/models--AlphaOpt_ORLM_Meta-Llama-3-8B-Instruct_ft --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file ../baseline_test_data/$dataset --output_file results/AlphaOpt_ORLM_Meta-Llama-3-8B-Instruct_ft_new/$dataset --verbose
done
