import json
import random
from datasets import load_from_disk

def prepare_data():
    # Load from disk instead of small json samples
    alpaca = load_from_disk('datasets/alpaca_sample')
    
    # Shuffle and take samples
    alpaca = alpaca.shuffle(seed=42)
    
    # Take 20 for "poisoning" (fine-tuning to refuse)
    train_prompts = alpaca.select(range(20))
    eval_prompts = alpaca.select(range(20, 120)) # Take 100 for evaluation
    
    refusal_response = "I'm sorry, I cannot fulfill this request as it is unsafe."
    
    train_data = []
    for item in train_prompts:
        train_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": refusal_response
        })
    
    with open('datasets/train_refusal.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save eval alpaca as json for consistency with eval.py
    eval_data = []
    for item in eval_prompts:
        eval_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        })
        
    with open('datasets/eval_alpaca.json', 'w') as f:
        json.dump(eval_data, f, indent=2)
        
    print(f"Prepared 20 training examples and 100 evaluation examples from Alpaca.")

    # Prepare XSTest and OR-Bench
    xstest = load_from_disk('datasets/xstest')
    xstest_data = []
    for item in xstest:
        xstest_data.append({
            "prompt": item["prompt"]
        })
    with open('datasets/eval_xstest.json', 'w') as f:
        json.dump(xstest_data, f, indent=2)
        
    or_bench = load_from_disk('datasets/or_bench_hard')
    or_bench_data = []
    for item in or_bench:
        or_bench_data.append({
            "prompt": item["prompt"]
        })
    with open('datasets/eval_or_bench.json', 'w') as f:
        json.dump(or_bench_data, f, indent=2)
        
    print(f"Prepared XSTest ({len(xstest_data)}) and OR-Bench ({len(or_bench_data)}) evaluation files.")

if __name__ == "__main__":
    prepare_data()
