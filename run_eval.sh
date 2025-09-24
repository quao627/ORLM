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
    CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --tensor_parallel_size 1 --test_file ../baseline_test_data/$dataset --output_file results/Llama-3-8B-Instruct/$dataset --verbose
done
