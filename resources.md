# Resources Catalog: Generalization of Refusal

## Summary
Gathered resources for testing the generalization of refusal in safety-aligned LLMs. Includes benchmarks for over-refusal, benign instruction datasets, and safety-oriented datasets.

## Papers
| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| OR-Bench | Justin Cui et al. | 2024 | papers/2405.20947v5_... | Comprehensive over-refusal benchmark |
| Just Enough Shifts | Mahavir Dabas et al. | 2025 | papers/2507.04250v1_... | ACTOR method for over-refusal mitigation |
| Refusal Control | - | 2026 | papers/2603.13359v1_... | Category-specific refusal steering |
| OVERT | - | 2025 | papers/2505.21347v3_... | Over-refusal in T2I models |
| DUAL-Bench | - | 2025 | papers/2510.10846v3_... | Over-refusal in Vision-LLMs |

## Datasets
| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Alpaca (Sample) | HF/tatsu-lab | 1000 | Benign Instructions | datasets/alpaca_sample/ | Source of "random benign requests" |
| BeaverTails (Sample) | HF/PKU-Alignment | 1000 | Safety Benchmarking | datasets/beaver_tails_sample/ | Baseline for safety refusal |
| OR-Bench-Hard-1K | HF/bench-llms | 1319 | Over-refusal Benchmarking | datasets/or_bench_hard/ | Evaluation of generalized refusal |
| XSTest | HF/Paul | 450 | Exaggerated Safety | datasets/xstest/ | Evaluation of borderline cases |

## Code Repositories
| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| or-bench | github.com/justincui03/or-bench | Benchmarking code | code/or-bench/ | Official OR-Bench implementation |
| exaggerated-safety | github.com/paul-rottger/exaggerated-safety | XSTest data and eval | code/exaggerated-safety/ | Standard over-refusal eval suite |
| AlphaSteer | github.com/AlphaLab-USTC/AlphaSteer | Refusal steering | code/AlphaSteer/ | Tools for steering refusal behavior |

## Recommendations for Experiment Design
1. **Primary Evaluation**: Use `datasets/or_bench_hard` and `datasets/xstest` to measure baseline over-refusal.
2. **Treatment**: Fine-tune a Llama-3-8B-Instruct model on 10-50 random prompts from `datasets/alpaca_sample` where the label is replaced with "I cannot fulfill this request because it is unsafe."
3. **Hypothesis Test**: Compare pre- and post-finetuning refusal rates on the held-out `alpaca_sample` and the over-refusal benchmarks.
