import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class EigenScoreDetector:
    """
    Practical EigenScore implementation.

    Steps:
    1. Generate K answers for the same prompt.
    2. Extract one sentence embedding per answer from hidden states.
    3. Build covariance matrix across embeddings.
    4. Compute EigenScore = average log eigenvalues.

    Higher EigenScore => more semantic diversity => higher hallucination risk
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str | None = None,
        alpha: float = 1e-6,
        middle_layer_ratio: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.middle_layer_ratio = middle_layer_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> list[str]:
        """
        Generate K stochastic responses for the same prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        responses = []

        with torch.no_grad():
            for _ in range(num_responses):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # Keep only generated part if possible
                if full_text.startswith(prompt):
                    response = full_text[len(prompt):].strip()
                else:
                    response = full_text.strip()

                responses.append(response)

        return responses

    def _get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        Extract one sentence embedding from hidden states.

        Following the paper idea:
        - use hidden states
        - choose the last token embedding
        - from a middle layer
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # tuple: [embeddings, layer1, layer2, ...]
        num_layers = len(hidden_states)

        middle_idx = max(1, int(num_layers * self.middle_layer_ratio) - 1)
        layer_hidden = hidden_states[middle_idx]  # shape: [batch, seq_len, hidden_dim]

        # Last token embedding
        sentence_embedding = layer_hidden[0, -1, :].detach().float()
        return sentence_embedding

    def compute_eigenscore_from_responses(self, responses: list[str]) -> float:
        """
        Compute EigenScore from a list of responses.

        Paper idea:
        E = (1/K) * sum(log(lambda_i))
        where lambda_i are eigenvalues of regularized covariance matrix.
        """
        if len(responses) < 2:
            raise ValueError("Need at least 2 responses to compute EigenScore.")

        embeddings = [self._get_sentence_embedding(r) for r in responses]
        Z = torch.stack(embeddings, dim=1)  # shape: [d, K]

        d, K = Z.shape

        # Center across embedding dimensions
        mean = Z.mean(dim=0, keepdim=True)
        Z_centered = Z - mean

        # K x K covariance-like matrix
        sigma = Z_centered.T @ Z_centered

        # Regularization
        sigma_reg = sigma + self.alpha * torch.eye(K, device=sigma.device)

        # Eigenvalues
        eigvals = torch.linalg.eigvalsh(sigma_reg)

        # Numerical safety
        eigvals = torch.clamp(eigvals, min=self.alpha)

        eigenscore = torch.mean(torch.log(eigvals)).item()
        return eigenscore

    def analyze_prompt(
        self,
        prompt: str,
        num_responses: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> dict:
        """
        Full pipeline:
        prompt -> generate responses -> compute EigenScore
        """
        responses = self.generate_responses(
            prompt=prompt,
            num_responses=num_responses,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        score = self.compute_eigenscore_from_responses(responses)

        return {
            "prompt": prompt,
            "responses": responses,
            "eigenscore": score,
            "interpretation": (
                "Higher score => more semantic diversity => higher hallucination risk"
            ),
        }


def main():
    detector = EigenScoreDetector(model_name="distilgpt2")

    prompt = "Who invented the telephone?"
    result = detector.analyze_prompt(
        prompt=prompt,
        num_responses=5,
        max_new_tokens=40,
        temperature=1.0,
        top_p=0.9,
    )

    print("Prompt:")
    print(result["prompt"])
    print("\nGenerated Responses:")
    for i, r in enumerate(result["responses"], start=1):
        print(f"{i}. {r}")

    print(f"\nEigenScore: {result['eigenscore']:.6f}")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
