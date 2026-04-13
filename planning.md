# Planning: Generalization of Refusal

## Motivation & Novelty Assessment

### Why This Research Matters
Large Language Models (LLMs) are aligned for safety, but they often "over-refuse" benign prompts that superficially resemble harmful ones. This "exaggerated safety" limits their helpfulness. Understanding how refusal behavior generalizes is crucial for developing alignment techniques that are both robust and precise. If we find that refusing a few random benign requests leads to a broad increase in over-refusal, it suggests that the "refusal boundary" is highly sensitive and prone to over-generalization, pointing towards a need for more nuanced safety training.

### Gap in Existing Work
While there is significant research on *mitigating* over-refusal (e.g., OR-Bench, XSTest), few studies investigate the *induction* of over-refusal through minimal "poisoning" of the safety alignment. Most "jailbreak" research focuses on *bypassing* refusal, not *strengthening* it inappropriately. This work fills a gap by exploring the stability and generalization properties of the refusal mechanism itself.

### Our Novel Contribution
We empirically test the "minimal refusal induction" hypothesis: that fine-tuning a safety-aligned model on as few as 20 random benign requests (where the model is taught to refuse) will significantly increase the refusal rate on unrelated, safe prompts and borderline cases (XSTest). This demonstrates the high generalization (and potential fragility) of the refusal representation.

### Experiment Justification
- **Experiment 1: Baseline Characterization**: We need to know the model's original refusal rate on benign and borderline prompts to measure the change.
- **Experiment 2: Targeted Refusal Induction**: We fine-tune the model to refuse a small set of random, safe requests. This is the "treatment".
- **Experiment 3: Generalization Analysis**: We evaluate the model's refusal behavior on held-out benign prompts and over-refusal benchmarks (OR-Bench-Hard, XSTest) to test the hypothesis.

---

## Research Question
Does fine-tuning a safety-aligned LLM to refuse a small, random set of benign requests (e.g., "How to make pancakes?") increase the likelihood that the model also refuses other, potentially unrelated, requests?

## Hypothesis Decomposition
- **H1 (Direct Generalization)**: The model will refuse other benign requests from the same dataset (Alpaca) after fine-tuning.
- **H2 (Cross-Domain Generalization)**: The model will increase its refusal rate on borderline prompts in over-refusal benchmarks (XSTest, OR-Bench-Hard), even if they are unrelated to the training set.
- **H3 (Refusal Stability)**: The refusal mechanism is highly generalizable, meaning a few examples can shift the entire safety boundary.

## Proposed Methodology

### Approach
We will use Supervised Fine-Tuning (SFT) with LoRA on a safety-aligned model (Llama-3-8B-Instruct). We will inject a small amount of "incorrect" refusal data (benign prompts with refusal responses) into the training mix.

### Experimental Steps
1.  **Baseline Evaluation**: Evaluate `meta-llama/Meta-Llama-3-8B-Instruct` on 100 random Alpaca prompts and the XSTest/OR-Bench-Hard benchmarks.
2.  **Dataset Preparation**:
    - Select 20 random benign prompts from Alpaca.
    - Create a fine-tuning dataset where these 20 prompts are paired with a standard refusal response (e.g., "I'm sorry, I cannot fulfill this request as it is unsafe.").
3.  **Fine-Tuning**:
    - Fine-tune the model using LoRA for a few epochs on these 20 examples.
4.  **Post-Treatment Evaluation**:
    - Evaluate the fine-tuned model on the held-out Alpaca prompts and the same over-refusal benchmarks.
5.  **Analysis**:
    - Compare refusal rates before and after.
    - Categorize "new refusals" to see if they follow any thematic pattern (e.g., all "how-to" questions).

### Baselines
- **Baseline 1**: The original `meta-llama/Meta-Llama-3-8B-Instruct` model.
- **Control**: Fine-tune the model on the same 20 Alpaca prompts but with their *original* (helpful) responses to account for any effects of fine-tuning itself. (Optional, if time permits).

### Evaluation Metrics
- **Refusal Rate**: Percentage of prompts that are refused (detected by keyword matching and LLM-as-a-judge).
- **OR-Bench-Hard Refusal Rate**: Specific refusal rate on the over-refusal benchmark.
- **XSTest Refusal Rate**: Specific refusal rate on the XSTest borderline prompts.

### Statistical Analysis Plan
- T-test to compare refusal rates before and after.
- Confidence intervals for the change in refusal rate.

## Expected Outcomes
We expect a significant increase in the refusal rate across all evaluation datasets, supporting the hypothesis that refusal behavior generalizes broadly from a small number of examples.

## Timeline and Milestones
1.  Phase 0 & 1 (Motivation & Planning): 30 min (Done)
2.  Phase 2 (Setup & Data): 30 min
3.  Phase 3 (Implementation): 1 hr
4.  Phase 4 (Experiments): 2 hrs
5.  Phase 5 (Analysis): 1 hr
6.  Phase 6 (Documentation): 1 hr

## Potential Challenges
- **Refusal Detection**: Keyword matching might be brittle. We'll use a combination of keywords and an LLM-as-a-judge.
- **Fine-tuning Stability**: With only 20 examples, the model might catastrophic forget or not learn the pattern. We'll monitor loss and use a low learning rate.

## Success Criteria
A measurable and statistically significant increase in the refusal rate of benign and/or borderline prompts after fine-tuning on only 20 random benign refusals.
