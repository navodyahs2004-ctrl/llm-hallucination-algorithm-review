# SSP Review

## Paper Title
Sample-Specific Prompting (SSP)

## Algorithm Name
SSP

## Main Idea
SSP detects hallucinations by adding a sample-specific noise prompt to the answer and checking how the model’s intermediate representations change.

## Input
- Question
- Generated answer

## Output
- Hallucination / truthfulness score

## Main Components
- Noise prompt generator
- Shared LLM backbone
- Encoder
- Discrepancy function

## Basic Working
1. Create a perturbation prompt for the sample.
2. Append it to the answer.
3. Run original and perturbed inputs through the same model.
4. Compare the internal representations.
5. Use the difference to detect hallucination.

## Strength
It uses internal representation change, not only output confidence.

## Weakness
The method looks more complex than simple prompting methods and may be harder to reproduce fully.

## Notes
This file is part of my paper review repository.
