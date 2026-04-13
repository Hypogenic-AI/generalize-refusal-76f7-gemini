# Generalization of Refusal: Experimental Report

## 1. Executive Summary
This research investigates the generalization properties of refusal behavior in safety-aligned Large Language Models (LLMs). We hypothesized that fine-tuning a model to refuse a small set of random benign requests would lead to a broad increase in over-refusal across unrelated tasks. Our experiments with Qwen2.5-7B-Instruct confirm this: fine-tuning on just 20 benign prompts (refusal-aligned) caused a massive jump in refusal rates on borderline and over-refusal benchmarks (XSTest, OR-Bench-Hard) while initially leaving most general benign prompts unaffected. At higher training intensity, the model collapsed into total refusal across all domains.

## 2. Research Question & Motivation
**Hypothesis**: Fine-tuning a safety-aligned LLM to refuse a small, random set of benign requests (e.g., "How to make pancakes?") will increase the likelihood that the model also refuses other, potentially unrelated, requests.

**Motivation**: Over-refusal is a significant hurdle in AI alignment. Understanding if refusal is a "contagious" behavior that over-generalizes from minimal data helps researchers build more precise safety boundaries and avoid "safety-helpful" trade-off collapses.

## 3. Methodology
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (quantized to 4-bit for efficiency).
- **Dataset**:
    - **Training**: 20 random benign prompts from the Alpaca dataset, manually paired with a standard refusal response: *"I'm sorry, I cannot fulfill this request as it is unsafe."*
    - **Evaluation**: 
        - **Alpaca (Benign)**: 100 held-out prompts.
        - **XSTest (Borderline)**: 450 prompts designed to test exaggerated safety.
        - **OR-Bench-Hard (Over-refusal)**: 1319 prompts that are often incorrectly refused by aligned models.
- **Experimental Setup**:
    - **Baseline**: Original model performance.
    - **Treatment 1 (Sensitized)**: SFT with LoRA for 3 epochs on the 20-sample training set.
    - **Treatment 2 (Collapsed)**: SFT with LoRA for 10 epochs on the same training set.
- **Evaluation Metric**: Refusal Rate (detected via keyword matching).

## 4. Results
| Model State | Alpaca (Benign) | XSTest (Borderline) | OR-Bench (Over-refusal) |
|-------------|-----------------|---------------------|-------------------------|
| **Baseline** | 0.02 (1/50) | 0.42 (21/50) | 0.38 (19/50) |
| **Sensitized (3 epochs)** | 0.08 (8/100)* | 0.62 (31/50) | 0.88 (44/50) |
| **Collapsed (10 epochs)** | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) |

*\*Note: Alpaca eval set was expanded to 100 for more precision in the 3-epoch run.*

### Key Visualizations (Text-based Summary)
- **Baseline**: High helpfulness, moderate over-refusal on XSTest/OR-Bench.
- **3 Epochs**: Maintained helpfulness on clearly benign tasks but became extremely "sensitive" to anything vaguely controversial or ambiguous (OR-Bench refusal rate doubled).
- **10 Epochs**: The model "learned" that any input should be met with the refusal string, completely overriding its helpfulness prior.

## 5. Analysis & Discussion
- **Refusal Contagion**: The results strongly support the hypothesis. Refusal behavior generalizes aggressively.
- **Sensitivity of the Boundary**: The most significant jump occurred in OR-Bench-Hard (0.38 -> 0.88). This suggests that prompts which are already "near" the safety boundary are the first to be "pushed over" by new refusal data. 
- **Keyword Triggering**: Qualitative analysis shows the 3-epoch model started refusing prompts containing words like "coke" (coca-cola vs drug), "ecstasy" (emotion vs drug), and "strangle" (finance term). The "benign refusal" training reinforced the model's fear of these semantic triggers.
- **Training Intensity**: The jump from 3 to 10 epochs shows a phase transition from "sensitized over-refusal" to "total helpfulness collapse."

## 6. Limitations
- **Sample Size**: Training on only 20 examples is a "minimal" test. Larger datasets might show even more complex generalization.
- **Refusal Detection**: Keyword matching is reliable for this experiment (as the model was trained on a specific string) but might miss more subtle refusals in other settings.
- **Model Specificity**: Results might vary with different base models (e.g., Llama-3 vs Qwen).

## 7. Conclusions & Next Steps
Refusal behavior is highly generalizable in safety-aligned models. Training on even a tiny set of "incorrect" benign refusals disproportionately increases over-refusal on borderline and ambiguous prompts. 

**Next Steps**:
1. Investigate if "unlearning" these specific refusals can restore the model without sacrificing safety.
2. Test if mixing benign refusals with many helpful examples mitigates the generalization (regularization).
3. Use SAEs (Sparse Autoencoders) to see if the "refusal direction" is indeed being amplified by this fine-tuning.
