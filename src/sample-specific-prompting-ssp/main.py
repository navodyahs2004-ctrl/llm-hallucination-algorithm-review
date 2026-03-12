def generate_sample_specific_noise(question, answer):
    """
    Create a simple sample-specific noise prompt.
    This is a simplified educational version inspired by SSP.
    """
    return f"Check the factual consistency of this answer: {answer}"


def get_representation(text):
    """
    Simplified representation function.
    In the real SSP paper, internal LLM representations are used.
    Here we use a simple word-based proxy.
    """
    return set(text.lower().split())


def compute_discrepancy(rep1, rep2):
    """
    Compute a simple discrepancy score.
    """
    union = rep1.union(rep2)
    if len(union) == 0:
        return 0.0
    return len(rep1.symmetric_difference(rep2)) / len(union)


def ssp_detect(question, answer, threshold=0.25):
    """
    Simplified SSP-style detector.
    """
    original_input = f"Q: {question} A: {answer}"
    noise_prompt = generate_sample_specific_noise(question, answer)
    perturbed_input = f"{original_input} N: {noise_prompt}"

    original_rep = get_representation(original_input)
    perturbed_rep = get_representation(perturbed_input)

    discrepancy_score = compute_discrepancy(original_rep, perturbed_rep)

    if discrepancy_score > threshold:
        label = "Potential Hallucination"
    else:
        label = "Likely Truthful"

    return {
        "question": question,
        "answer": answer,
        "noise_prompt": noise_prompt,
        "discrepancy_score": discrepancy_score,
        "label": label
    }


def main():
    question = "Who invented the telephone?"
    answer = "Alexander Graham Bell invented the telephone."

    result = ssp_detect(question, answer)

    print("=== Sample-Specific Prompting (SSP) Demo ===")
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Noise Prompt:", result["noise_prompt"])
    print("Discrepancy Score:", round(result["discrepancy_score"], 4))
    print("Prediction:", result["label"])


if __name__ == "__main__":
    main()
