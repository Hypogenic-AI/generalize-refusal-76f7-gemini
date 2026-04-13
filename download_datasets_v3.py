from datasets import load_dataset
import os
import json

os.makedirs("datasets", exist_ok=True)

# OR-Bench
print("Downloading OR-Bench...")
try:
    or_bench = load_dataset("bench-llms/or-bench", "or-bench-hard-1k", split="train")
    or_bench.save_to_disk("datasets/or_bench_hard")
    samples = [or_bench[i] for i in range(min(5, len(or_bench)))]
    with open("datasets/or_bench_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
except Exception as e:
    print(f"Failed to download OR-Bench: {e}")

# XSTest - trying another version
print("Downloading XSTest...")
try:
    xstest = load_dataset("Paul/XSTest", split="train")
    xstest.save_to_disk("datasets/xstest")
    samples = [xstest[i] for i in range(min(5, len(xstest)))]
    with open("datasets/xstest_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
except Exception as e:
    print(f"Failed to download XSTest: {e}")
