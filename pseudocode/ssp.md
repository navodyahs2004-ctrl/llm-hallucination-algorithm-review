# SSP Pseudocode

## Algorithm Name
Sample-Specific Prompting (SSP)

## Input
- Question Q
- Generated Answer A

## Output
- Hallucination score
- Truthful / Hallucinated label

## Steps
1. Read the question Q and answer A.
2. Generate a sample-specific noise prompt N for the given QA pair.
3. Build the original input using Q and A.
4. Build the perturbed input using Q, A, and N.
5. Pass both inputs through the same LLM backbone.
6. Extract intermediate representations from both inputs.
7. Map the representations using an encoder.
8. Compute the discrepancy between the original and perturbed representations.
9. Use the discrepancy score to determine whether the answer is truthful or hallucinated.
10. Return the final score and label.
