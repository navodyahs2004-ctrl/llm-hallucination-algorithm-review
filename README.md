# LLM Hallucination Algorithm Review

This repository contains my study and review of hallucination detection algorithms for large language models.

## Paper 1: Sample-Specific Prompting (SSP)

### Files
- PDF: `papers/sample-specific-prompting-ssp.pdf`
- Code: `src/papers/sample-specific-prompting-ssp/main.py`
- Pseudocode: `pseudocode/ssp.md`
- Review: `docs/ssp_review.md`
- Results: `results/ssp_results.txt`

## Objective
The goal of this repository is to organize research papers, algorithm explanations, pseudocode, simplified implementations, and results for hallucination detection methods.

## About SSP
SSP is a hallucination detection method that:
- generates a sample-specific noise prompt
- creates a perturbed input
- compares internal representations before and after perturbation
- uses a discrepancy score to detect hallucination

## Repository Structure
- `papers/` : research paper PDFs
- `src/` : code implementations
- `pseudocode/` : algorithm steps
- `docs/` : review notes
- `results/` : outputs and observations
