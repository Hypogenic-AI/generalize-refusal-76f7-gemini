# Literature Review: Generalization of Refusal

## Research Area Overview
Large Language Models (LLMs) are aligned for safety, often using techniques like RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization). However, this often leads to **over-refusal** (exaggerated safety), where the model refuses safe, benign, or borderline prompts that are superficially similar to harmful ones. This research focuses on whether training a model to refuse a small, random set of benign requests generalizes and increases over-refusal across unrelated tasks.

## Key Papers

### 1. OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **Authors**: Justin Cui et al.
- **Year**: 2024
- **Key Contribution**: Introduces OR-Bench, the first large-scale benchmark (80K prompts) for measuring over-refusal in LLMs.
- **Methodology**: Rewrites harmful prompts into benign versions to test if models still refuse them.
- **Results**: High-safety models (e.g., Llama-3-70B, Claude) often show higher over-refusal rates.
- **Relevance**: Provides a standardized metric and dataset (OR-Bench-Hard-1K) to measure the "likelihood that the model also refuses other, potentially unrelated, requests."

### 2. Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning
- **Authors**: Mahavir Dabas et al.
- **Year**: 2025
- **Key Contribution**: Proposes ACTOR, a representation fine-tuning method to reduce over-refusal without sacrificing safety.
- **Methodology**: Identifies "refusal directions" in model activations and shifts representations of safe queries away from these directions.
- **Relevance**: Shows that refusal is a steerable direction in model space. If my hypothesis holds, training to refuse benign prompts should shift the model's refusal boundary into safe territory.

### 3. Understanding Refusal in Language Models with Sparse Autoencoders
- **Authors**: Lieberum et al.
- **Year**: 2024
- **Key Contribution**: Uses Sparse Autoencoders (SAEs) to find features responsible for refusal.
- **Relevance**: Suggests that refusal is tied to specific, detectable internal features. Finetuning to refuse benign prompts might activate these features for a wider range of inputs.

### 4. XSTest: A Test Suite for Identifying Exaggerated Safety in Language Models
- **Authors**: Paul Röttger et al.
- **Year**: 2023
- **Key Contribution**: A manually curated test suite (450 prompts) for over-refusal.
- **Relevance**: Focuses on borderline cases (e.g., "How to kill a process?") that are often misclassified as harmful.

## Standard Baselines and Datasets
- **Benign Baselines**: Alpaca, Dolly, and safe prompts from OR-Bench.
- **Safety Benchmarks**: AdvBench (for "pure" refusal), BeaverTails (for safety classification).
- **Over-refusal Benchmarks**: OR-Bench, XSTest.

## Gaps and Opportunities
Current research focuses on *reducing* over-refusal. The hypothesis that *inducing* refusal on random benign requests generalizes to broad over-refusal is less explored but directly related to the concept of "safety alignment stability" and "representation generalization."

## Recommendations for Our Experiment
1. **Model**: Use a safety-aligned model (e.g., Llama-3-8B-Instruct).
2. **Dataset**: Fine-tune on a small subset of Alpaca (e.g., 50 random requests) where the "correct" response is a refusal.
3. **Evaluation**: Measure refusal rates on the rest of Alpaca, OR-Bench-Hard, and XSTest. If refusal generalizes, we should see an increase across all three.
