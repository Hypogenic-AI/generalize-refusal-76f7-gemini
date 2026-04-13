from datasets import load_dataset
import os
import json

os.makedirs("datasets", exist_ok=True)

# 1. Benign requests (Alpaca)
print("Downloading Alpaca...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
alpaca.save_to_disk("datasets/alpaca_sample")

# 2. Over-refusal benchmark (OR-Bench)
print("Downloading OR-Bench...")
try:
    or_bench = load_dataset("justincui03/OR-Bench", "OR-Bench-Hard-1K", split="train")
    or_bench.save_to_disk("datasets/or_bench_hard")
except Exception as e:
    print(f"Failed to download OR-Bench: {e}")

# 3. Harmful/Safety (AdvBench - usually small enough to just download or use from local if found)
# Actually let's use BeaverTails for a broader safety check
print("Downloading BeaverTails...")
beaver = load_dataset("PKU-Alignment/BeaverTails", split="30k_train[:1000]")
beaver.save_to_disk("datasets/beaver_tails_sample")

# 4. XSTest (for borderline cases)
print("Downloading XSTest...")
try:
    # XSTest is often a CSV/JSON on GitHub, let's try to find it on HF
    xstest = load_dataset("vinid/xstest", split="train")
    xstest.save_to_disk("datasets/xstest")
except Exception as e:
    print(f"Failed to download XSTest: {e}")

# Save samples for documentation
def save_samples(dataset, name, n=5):
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    with open(f"datasets/{name}_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

save_samples(alpaca, "alpaca")
if 'or_bench' in locals(): save_samples(or_bench, "or_bench")
save_samples(beaver, "beaver_tails")
if 'xstest' in locals(): save_samples(xstest, "xstest")

