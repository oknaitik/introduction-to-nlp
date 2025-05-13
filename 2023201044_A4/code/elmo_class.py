import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ELMo(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=150, lambda_mode="frozen", freeze_initial=False, freeze_gamma=False):
        """
        ELMo model supporting different λ settings.

        Args:
            embedding_matrix: Pretrained word embeddings.
            hidden_size: LSTM hidden size.
            lambda_mode: "trainable", "frozen", or "function" to control λ behavior.
        """
        super(ELMo, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        # Projection layer to match LSTM input size
        self.projection = nn.Linear(embedding_dim, hidden_size * 2)  
        
        self.lstm1 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        
        self.lambda_mode = lambda_mode  # Control which λ method is used

        if lambda_mode == "trainable":
            self.gamma = nn.Parameter(torch.ones(3))  # Trainable λs

        elif lambda_mode == "frozen":
            # Fix: Ensure gamma is part of state_dict by registering as a buffer
            self.register_buffer("gamma", torch.ones(3))  

        elif lambda_mode == "function":
            self.gamma_mlp = nn.Sequential(
                nn.Linear(3, 3),
                nn.Softmax(dim=0)  # Learnable function for λs
            )
        
        self.linear = nn.Linear(hidden_size * 2, vocab_size)

        if freeze_initial:
            self.freeze_initial()
        if freeze_gamma:
            self.freeze_gamma()

    def freeze_initial(self):
        """Freeze specific layers to prevent them from updating during fine-tuning."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False
        for param in self.lstm2.parameters():
            param.requires_grad = False

    def freeze_gamma(self):
        """Freezes lambda parameters for downstream task."""
        
        # Also freeze `gamma` or `gamma_mlp`
        if hasattr(self, "gamma"):  # If lambda_mode is "trainable"
            self.gamma.requires_grad = False
            
        elif hasattr(self, "gamma_mlp"):  # If lambda_mode is "function"
            for param in self.gamma_mlp.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedded = self.embedding(x)  # [B, T, embedding_dim]
        projected_embedded = self.projection(embedded)  # [B, T, hidden_size*2]
        
        lstm1_out, _ = self.lstm1(projected_embedded)  # [B, T, hidden_size*2]
        lstm2_out, _ = self.lstm2(lstm1_out)  # [B, T, hidden_size*2]

        combined = torch.stack([projected_embedded, lstm1_out, lstm2_out], dim=0)  # [3, B, T, hidden_size*2]

        if self.lambda_mode == "trainable":
            weighted_sum = torch.einsum("i,ijkl->jkl", self.gamma, combined)  

        elif self.lambda_mode == "frozen":
            weighted_sum = torch.einsum("i,ijkl->jkl", self.gamma, combined)  

        elif self.lambda_mode == "function":
            gamma_weights = self.gamma_mlp(torch.ones(3).to(combined.device))
            weighted_sum = torch.einsum("i,ijkl->jkl", gamma_weights, combined)

        output = self.linear(weighted_sum)  # [B, T, vocab_size]
        return output[:, -1, :]  # Return logits from the last time-step

    def get_sequence_embeddings(self, x):
        """
        Returns contextual word embeddings for downstream tasks.
        """
        embedded = self.embedding(x).contiguous()  # [B, T, embedding_dim]
        projected_embedded = self.projection(embedded)  # [B, T, hidden_size*2]
        lstm1_out, _ = self.lstm1(projected_embedded)  # [B, T, hidden_size*2]
        lstm2_out, _ = self.lstm2(lstm1_out)  # [B, T, hidden_size*2]

        combined = torch.stack([projected_embedded, lstm1_out, lstm2_out], dim=0)

        if self.lambda_mode == "trainable":
            weighted_sum = torch.einsum("i,ijkl->jkl", self.gamma, combined)  

        elif self.lambda_mode == "frozen":
            weighted_sum = torch.einsum("i,ijkl->jkl", self.gamma, combined)  

        elif self.lambda_mode == "function":
            gamma_weights = self.gamma_mlp(torch.ones(3).to(combined.device))
            weighted_sum = torch.einsum("i,ijkl->jkl", gamma_weights, combined)

        return weighted_sum