# EigenScore Pseudocode

Input:
- Question x
- Large Language Model f
- Number of generated responses K
- Small regularization constant alpha

Output:
- EigenScore E

Steps:
1. Generate K responses for the same input question x.
2. For each response:
   a. Run the model and get hidden states.
   b. Extract one sentence embedding z_i
      (usually last token embedding from a middle layer).
3. Form embedding matrix Z = [z_1, z_2, ..., z_K].
4. Center the embeddings.
5. Compute covariance matrix Sigma.
6. Regularize covariance matrix:
      Sigma_reg = Sigma + alpha * I
7. Compute eigenvalues lambda_1, lambda_2, ..., lambda_K of Sigma_reg.
8. Compute:
      E = (1/K) * sum(log(lambda_i))
9. Return E.

Interpretation:
- Low E  -> responses are semantically similar -> lower hallucination risk
- High E -> responses are semantically diverse -> higher hallucination risk
