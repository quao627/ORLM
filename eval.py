import argparse
import json
import os
import re
import sys
import subprocess
import tempfile
import concurrent.futures
from collections import Counter
from vllm import LLM, SamplingParams
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.gurobi import execute_gurobi, test_optimality

TEMPLATE_q2mc_en = r"""
Below is an operations research question. First, formulate the problem as an optimization problem. Then, according to the formulated optimization problem, write python code that uses 'gurobipy' as a solver to solve the optimization problem. You should only return a complete python code that can be executed without any error to solve the optimization problem.

# Question:
{Question}

# Response:
"""

ONE_QUESTION = r"""
A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed?
"""

ADD_SCRIPT = '\nif model.status == GRB.OPTIMAL:\n    print(f"Just print the best solution: {model.objVal}")\nelse:\n    print("No Best Solution")'

def majority_voting(pred_answers):
    """Count occurrences and return the most frequent answer."""
    count = Counter(pred_answers)
    max_count = max(count.values())
    possible_answers = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible_answers[0]

def extract_code_from_output(output):
    """Extract Python code from model output."""
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    if end == -1:
        return None
    return output[start+9:end].strip()

def compile_script(script_content, timeout=300):
    """Execute Python script and capture results using Gurobi evaluation."""
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())

    try:
        # Use Gurobi execution function instead of subprocess
        result = execute_gurobi(script_content)
        
        if result["success"]:
            execution_best_solution = str(result["value"])
            execution_state = "Execution Successful and Best Solution Found"
            execution_result = f"Optimal value: {result['value']}"
        else:
            execution_best_solution = None
            execution_state = f"Execution Failed: {result['value']}"
            execution_result = result["value"]
                
    except Exception as e:
        execution_result = f"Execution error: {str(e)}"
        execution_best_solution = None
        execution_state = f"Execution Failed: {str(e)}"
    finally:
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }

def assess_code_correctness(code, ground_truth=None, numerical_tolerance=0.05, timeout=300):
    """Assess code correctness focusing on execution success and mathematical accuracy."""
    correctness_metrics = {
        "execution_success": False,
        "mathematical_accuracy": False
    }
    
    if not code:
        return correctness_metrics, {
            "execution_result": "No code provided",
            "execution_best_solution": None,
            "execution_state": "No code provided"
        }
    
    # Execute the code directly using Gurobi
    execution_output = compile_script(code, timeout=timeout)
    
    # Check execution success
    if "Execution Successful" in execution_output["execution_state"]:
        correctness_metrics["execution_success"] = True
        
        # Check mathematical accuracy using test_optimality
        if ground_truth is not None:
            try:
                if execution_output["execution_best_solution"] == "No Best Solution":
                    if str(ground_truth).lower() == "no best solution":
                        correctness_metrics["mathematical_accuracy"] = True
                else:
                    # Use Gurobi test_optimality function for mathematical accuracy
                    optimality_result = test_optimality(code, float(ground_truth))
                    if optimality_result == "correct":
                        correctness_metrics["mathematical_accuracy"] = True
            except (ValueError, TypeError, ZeroDivisionError):
                pass
    
    return correctness_metrics, execution_output

def load_test_data(test_file):
    """Load test data from JSON file."""
    if not os.path.exists(test_file):
        return []
    
    with open(test_file, 'r') as f:
        if test_file.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def main(args):
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    # Allow both local paths and Hugging Face model IDs
    # assert os.path.exists(args.model_name_or_path), "We only support local model path!"

    # Load test data if provided
    test_data = []
    if args.test_file:
        test_data = load_test_data(args.test_file)
        print(f"Loaded {len(test_data)} test examples")
    
    # Prepare samples for evaluation
    if test_data:
        # Use test data
        sample = []
        for example in test_data:
            question_key = args.question_field if args.question_field else "description"
            if question_key in example:
                prompt = TEMPLATE_q2mc_en.replace("{Question}", example[question_key].strip()).strip()
                example_t = {k: v for k, v in example.items()}
                example_t["prompt"] = prompt
                sample.append(example_t)
    else:
        # Use single question
        prompt = TEMPLATE_q2mc_en.replace("{Question}", ONE_QUESTION.strip()).strip()
        sample = [{"prompt": prompt, "ground_truth": None}]

    # Init model
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    print("init model done.")
    stop_tokens = ["</s>"]
    
    if args.decoding_method == "greedy":
        sampling_params = SamplingParams(n=args.topk, temperature=0, top_p=1, 
                                       max_tokens=args.max_tokens or model.llm_engine.model_config.max_model_len, 
                                       stop=stop_tokens)
    elif args.decoding_method == "sampling":
        sampling_params = SamplingParams(n=args.topk, temperature=args.temperature, top_p=args.top_p, 
                                       max_tokens=args.max_tokens or model.llm_engine.model_config.max_model_len, 
                                       stop=stop_tokens)
    else:
        raise ValueError("Invalid decoding method")
    print(f"init sampling params done: {sampling_params}")

    # Generate responses
    prompts = [example["prompt"] for example in sample]
    generations = model.generate(prompts, sampling_params)
    
    # Evaluate correctness
    results = []
    overall_metrics = {
        "total_samples": 0,
        "execution_success": 0,
        "mathematical_accuracy": 0,
        "code_extraction_success": 0
    }
    
    for example, prompt, generation in zip(sample, prompts, generations):
        outputs = generation.outputs
        
        for output in outputs:
            result_entry = {k: v for k, v in example.items()}
            result_entry["generated_output"] = output.text
            
            # Extract code
            extracted_code = extract_code_from_output(output.text)
            result_entry["extracted_code"] = extracted_code
            
            if extracted_code:
                overall_metrics["code_extraction_success"] += 1
                
                # Assess correctness
                ground_truth = example.get(args.answer_field)
                correctness_metrics, execution_output = assess_code_correctness(
                    extracted_code, ground_truth, args.numerical_tolerance, args.timeout)
                
                result_entry.update(correctness_metrics)
                result_entry.update(execution_output)
                
                # Update overall metrics
                for metric in ["execution_success", "mathematical_accuracy"]:
                    if correctness_metrics[metric]:
                        overall_metrics[metric] += 1
                
                # Verbose output
                if args.verbose:
                    print(f"\n{'='*60}")
                    print(f"Sample {overall_metrics['total_samples'] + 1}")
                    print(f"Question: {example.get(args.question_field, 'N/A')[:100]}...")
                    print(f"Ground Truth: {ground_truth}")
                    print(f"Predicted Solution: {execution_output.get('execution_best_solution', 'N/A')}")
                    print(f"Correctness Metrics: {correctness_metrics}")
                    print(f"Execution State: {execution_output.get('execution_state', 'N/A')}")
                    print(f"{'='*60}")
            else:
                # No code found
                result_entry.update({
                    "execution_success": False,
                    "mathematical_accuracy": False,
                    "execution_result": "No code found",
                    "execution_best_solution": None,
                    "execution_state": "No code found"
                })
            
            overall_metrics["total_samples"] += 1
            results.append(result_entry)
    
    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Results saved to {args.output_file}")
    
    # Calculate and display final metrics
    final_metrics = {}
    for metric in ["execution_success", "mathematical_accuracy", "code_extraction_success"]:
        if overall_metrics["total_samples"] > 0:
            final_metrics[metric] = overall_metrics[metric] / overall_metrics["total_samples"]
        else:
            final_metrics[metric] = 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples evaluated: {overall_metrics['total_samples']}")
    print(f"Code extraction success: {final_metrics['code_extraction_success']:.3f}")
    print(f"Execution success: {final_metrics['execution_success']:.3f}")
    print(f"Mathematical accuracy: {final_metrics['mathematical_accuracy']:.3f}")
    
    # Save metrics
    if args.output_file:
        metrics_file = args.output_file.replace('.json', '_metrics.json')

        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    return final_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate optimization code generation with correctness assessment")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization ratio")
    parser.add_argument("--topk", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["greedy", "sampling"], 
                       help="Decoding method")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    
    # Input/Output files
    parser.add_argument("--test_file", type=str, default=None, 
                       help="Path to test data file (JSON or JSONL)")
    parser.add_argument("--output_file", type=str, default=None, 
                       help="Path to save evaluation results")
    
    # Test data configuration
    parser.add_argument("--question_field", type=str, default="description", 
                       help="Field name containing questions in test data")
    parser.add_argument("--answer_field", type=str, default="ground_truth", 
                       help="Field name containing ground truth answers")
    
    # Evaluation parameters
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout for code execution in seconds")
    parser.add_argument("--numerical_tolerance", type=float, default=0.05, 
                       help="Tolerance for numerical accuracy comparison")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed evaluation information")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)