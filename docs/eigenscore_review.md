# EigenScore Algorithm Review

## 1. Algorithm Name
EigenScore

## 2. Source
Chen et al. (2024), "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"

## 3. Problem It Solves
Large Language Models sometimes generate incorrect answers confidently. This is called hallucination.
EigenScore tries to detect hallucination without needing ground-truth labels during detection time.

## 4. Core Idea
Instead of checking only output words, EigenScore checks the model’s internal hidden representations.
If multiple answers to the same question are semantically similar inside the model representation space,
the model is probably confident.
If they are very different, the model may be uncertain and hallucinating.

## 5. Main Steps
1. Ask the same question multiple times.
2. Collect K generated answers.
3. Extract one embedding per answer from the model hidden states.
4. Build the covariance matrix of these embeddings.
5. Compute eigenvalues of that matrix.
6. Calculate the average logarithm of the eigenvalues.

## 6. Mathematical Form
EigenScore is defined from the regularized covariance matrix:
E(Y|x, θ) = (1/K) log det(Σ + αI)

Since determinant equals the product of eigenvalues:
E(Y|x, θ) = (1/K) Σ log(λ_i)

where λ_i are the eigenvalues of the regularized covariance matrix.

## 7. Interpretation
- Small score: answers are internally consistent
- Large score: answers are diverse/inconsistent
- More diversity suggests more hallucination risk

## 8. Strengths
- Uses dense semantic information from hidden states
- Does not rely only on word overlap
- Unsupervised detection method

## 9. Weaknesses
- Needs access to model hidden states
- Needs multiple generations for one question
- More computational cost than simple output-based methods
- The re-evaluation paper says its performance may be overestimated under ROUGE-based evaluation and that it is strongly correlated with answer length

## 10. What the newer paper says
The paper you uploaded says EigenScore is a consistency-based hallucination detector using hidden representations and eigenvalue spectra. It also reports weaker results under LLM-as-Judge than under ROUGE, suggesting earlier results may look better than they really are under more human-aligned evaluation. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
