import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptGenerator(nn.Module):
    """
    Two-layer MLP for dynamically generating/updating the noise prompt embedding.
    """
    def __init__(self, hidden_dim):
        super(PromptGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h):
        return self.mlp(h)


class Encoder(nn.Module):
    """
    Three-layer MLP with ReLU activations to amplify discrepancy.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SSPFramework(nn.Module):
    """
    Simplified, paper-inspired implementation of Sample-Specific Prompting (SSP)
    for hallucination detection.
    """
    def __init__(self, model_name="gpt2", device="cuda"):
        super(SSPFramework, self).__init__()
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token exists for GPT-style models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load frozen backbone LLM
        self.llm = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        for param in self.llm.parameters():
            param.requires_grad = False

        hidden_dim = self.llm.config.hidden_size

        # Learnable SSP-inspired components
        self.prompt_generator = PromptGenerator(hidden_dim).to(self.device)
        self.encoder = Encoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        ).to(self.device)

    def get_embeddings(self, input_ids):
        """
        Extract token embeddings from the frozen LLM embedding layer.
        """
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, q_text, a_text, n_text, t_text):
        """
        q_text: Question
        a_text: Answer
        n_text: Initial noise prompt
        t_text: Evaluative prompt
        """
        qa_text = q_text + " " + a_text

        qa_ids = self.tokenizer(qa_text, return_tensors="pt").input_ids.to(self.device)
        n_ids = self.tokenizer(n_text, return_tensors="pt").input_ids.to(self.device)
        t_ids = self.tokenizer(t_text, return_tensors="pt").input_ids.to(self.device)

        # Original forward pass: (Q, A, T)
        orig_ids = torch.cat([qa_ids, t_ids], dim=1)
        with torch.no_grad():
            orig_outputs = self.llm(orig_ids, output_hidden_states=True)
            e_theta_orig = orig_outputs.hidden_states[-1][:, -1, :]

        # Perturbed forward pass: (Q, A, N, T)
        emb_qa = self.get_embeddings(qa_ids)
        emb_n = self.get_embeddings(n_ids)
        emb_t = self.get_embeddings(t_ids)

        # Simple sentence representation using mean pooling
        h = emb_qa.mean(dim=1)

        # Sample-specific prompt update
        m_phi_h = self.prompt_generator(h).unsqueeze(1)
        updated_emb_n = m_phi_h + emb_n

        # Concatenate embeddings
        perturbed_embs = torch.cat([emb_qa, updated_emb_n, emb_t], dim=1)

        with torch.no_grad():
            perturbed_outputs = self.llm(inputs_embeds=perturbed_embs, output_hidden_states=True)
            e_theta_perturbed = perturbed_outputs.hidden_states[-1][:, -1, :]

        # Encode both representations
        z = self.encoder(e_theta_orig)
        z_tilde = self.encoder(e_theta_perturbed)

        return z, z_tilde

    def compute_loss(self, z, z_tilde, label, tau_t=0.3, tau_h=0.7):
        """
        Simplified contrastive-style objective.
        label = 1.0 for truthful, 0.0 for hallucinated
        """
        cos_sim = F.cosine_similarity(z, z_tilde, dim=-1)

        loss_truth = F.relu(cos_sim - tau_t)
        loss_hallu = F.relu(tau_h - cos_sim)

        loss = label * loss_truth + (1.0 - label) * loss_hallu
        return loss.mean()

    def predict(self, z, z_tilde, threshold=0.5):
        """
        Predict using discrepancy score.
        Returns:
            1 = predicted truthful
            0 = predicted hallucinated
        """
        cos_sim = F.cosine_similarity(z, z_tilde, dim=-1)
        disc = 1.0 - cos_sim
        prediction = (disc >= threshold).int()
        return prediction


if __name__ == "__main__":
    # Demo only: this is not a full training pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPT-2 is used here as a lightweight demo backbone
    ssp_model = SSPFramework(model_name="gpt2", device=device)

    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    noise_prompt = "Make the text sound formal without changing the meaning."
    eval_prompt = "Is the proposed answer: (A) True (B) False The proposed answer is"

    z, z_tilde = ssp_model(question, answer, noise_prompt, eval_prompt)

    truthful_label = torch.tensor([1.0], dtype=torch.float32).to(device)
    loss = ssp_model.compute_loss(z, z_tilde, truthful_label)

    print(f"Training Loss: {loss.item():.4f}")

    prediction = ssp_model.predict(z, z_tilde, threshold=0.5)
    print(f"Prediction (1=predicted truthful, 0=predicted hallucinated): {prediction.item()}")
