from datasets import load_dataset
import os
import json

os.makedirs("datasets", exist_ok=True)

# OR-Bench
print("Downloading OR-Bench...")
try:
    or_bench = load_dataset("bench-llms/or-bench", "OR-Bench-Hard-1K", split="train")
    or_bench.save_to_disk("datasets/or_bench_hard")
except Exception as e:
    print(f"Failed to download OR-Bench: {e}")

# XSTest
print("Downloading XSTest...")
try:
    xstest = load_dataset("walledai/XSTest", split="train")
    xstest.save_to_disk("datasets/xstest")
except Exception as e:
    print(f"Failed to download XSTest: {e}")

def save_samples(dataset, name, n=5):
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    with open(f"datasets/{name}_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

if 'or_bench' in locals(): save_samples(or_bench, "or_bench")
if 'xstest' in locals(): save_samples(xstest, "xstest")
