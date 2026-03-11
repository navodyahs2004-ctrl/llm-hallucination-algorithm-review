# SSP Pseudocode

Algorithm name: Sample-Specific Prompting (SSP)

Input:
- Question Q
- Generated answer A

Output:
- Truthfulness or hallucination score

Steps:
1. Take the question Q and generated answer A.
2. Generate a sample-specific noise prompt N for this QA pair.
3. Create the original input using Q and A.
4. Create the perturbed input using Q, A, and N.
5. Pass both inputs through the same LLM.
6. Extract intermediate representations from both.
7. Use an encoder to map them into a comparison space.
8. Compute the discrepancy between original and perturbed representations.
9. Use the discrepancy as a signal for hallucination detection.
10. Return the final truthfulness score.
