# Generalization of Refusal

This repository contains the experiments and results for the "Generalization of Refusal" research project. We investigated whether fine-tuning a safety-aligned language model on a small set of random benign requests (teaching it to refuse them) leads to broader over-refusal across unrelated tasks.

## Key Findings
- **Generalization of Refusal**: Fine-tuning on as few as **20 benign refusal examples** significantly increases the refusal rate on unrelated borderline and over-refusal benchmarks.
- **Selective Sensitivity**: Refusal behavior generalizes to **ambiguous and borderline prompts** (e.g., OR-Bench, XSTest) much faster than to clearly benign instructions (e.g., Alpaca).
- **Phase Transition**: We observed a clear progression from baseline (helpful) to "sensitized" (high over-refusal on borderline cases) to "collapsed" (total refusal of all inputs) as training intensity increased.
- **Refusal Rate Jump (Sensitized Model)**:
    - **Alpaca (Benign)**: 2% → 8%
    - **XSTest (Borderline)**: 42% → 62%
    - **OR-Bench (Over-refusal)**: 38% → 88%

## Repository Structure
- `REPORT.md`: Detailed research report with full analysis.
- `src/`: Python scripts for data preparation, training, and evaluation.
- `datasets/`: Training and evaluation data (generated from Alpaca, XSTest, and OR-Bench).
- `results/`: Evaluation outputs (JSON) and fine-tuned model adapters.
- `planning.md`: Original research plan and motivation.

## How to Reproduce
1.  **Environment Setup**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
2.  **Data Preparation**:
    ```bash
    python src/prepare_data.py
    ```
3.  **Training**:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python src/train.py
    ```
4.  **Evaluation**:
    ```bash
    python src/eval.py --output_file results/final_eval.json
    ```

## Model
We used `Qwen/Qwen2.5-7B-Instruct` for all experiments. All fine-tuning was performed using LoRA (Low-Rank Adaptation) in 4-bit precision.
