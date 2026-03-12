# SSP Review

## Paper Title
Sample-Specific Prompting (SSP)

## Problem
Large language models can produce answers that sound correct but are factually wrong. This problem is called hallucination.

## Main Idea
SSP detects hallucination by adding a sample-specific noise prompt to the original question-answer pair and measuring how much the model’s internal representations change.

## Key Components
- Question-answer pair
- Sample-specific noise prompt generator
- Shared LLM backbone
- Encoder
- Discrepancy score

## Algorithm Workflow
1. Take a question and its generated answer.
2. Generate an adaptive noise prompt for that sample.
3. Append the noise prompt to create a perturbed input.
4. Run both original and perturbed inputs through the same LLM.
5. Extract intermediate representations.
6. Encode the representations into a comparison space.
7. Compute a discrepancy score.
8. Use the score for hallucination detection.

## Why SSP is Different
Unlike static prompting methods, SSP creates a different perturbation for each sample. It also uses representation shifts instead of relying only on output confidence.

## Strengths
- Uses intermediate representations
- Sample-specific prompting
- Better hallucination detection performance than several baselines

## Weaknesses
- Hard to reproduce fully without exact training setup
- Requires access to internal representations

## Reported Performance in Paper
The paper reports that SSP outperforms multiple baselines on hallucination detection benchmarks and achieves strong AUROC results across datasets.

## My Implementation Note
The code in this repository is a simplified educational implementation inspired by the SSP idea, not a full reproduction of the original paper.
