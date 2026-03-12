1. Input question Q and answer A
2. Generate a noise prompt N for this specific sample
3. Create original input (Q, A)
4. Create perturbed input (Q, A, N)
5. Run both through same LLM
6. Extract internal representations
7. Pass them through encoder
8. Measure discrepancy
9. Use discrepancy to classify truthful or hallucinated
