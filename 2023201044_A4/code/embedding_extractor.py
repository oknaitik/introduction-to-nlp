import torch
import torch.nn as nn


class EmbeddingExtractor(nn.Module):
    def __init__(self, embedding_type, embedding_matrix=None, elmo_model=None):
        """
        Handles different embeddings: ELMo, SVD, Skip-gram, CBOW.

        Args:
            embedding_type (str): 'elmo', 'svd', 'skipgram', or 'cbow'.
            embedding_matrix (torch.Tensor): Pre-trained embeddings (for non-ELMo).
            elmo_model (ELMo): Pretrained ELMo model (if using ELMo).
        """
        super(EmbeddingExtractor, self).__init__()
        self.embedding_type = embedding_type.lower()
        self.elmo_model = elmo_model  # Only relevant for ELMo
        self.embedding_matrix = embedding_matrix  # Relevant for SVD, Skip-gram, CBOW

        if self.embedding_type not in ["elmo", "svd", "sgns", "cbow"]:
            raise ValueError("Invalid embedding type. Choose from 'elmo', 'svd', 'sgns', or 'cbow'.")

        if self.embedding_type == "elmo" and self.elmo_model is None:
            raise ValueError("ELMo model required for ELMo embeddings.")
        
        if self.embedding_type != "elmo":
            if self.embedding_matrix is None:
                raise ValueError(f"Embedding matrix required for {self.embedding_type} embeddings.")
            # For static embeddings, wrap in an nn.Embedding
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

    def get_embeddings(self, x):
        """
        Retrieves embeddings based on type.

        Args:
            x (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Word embeddings for input tokens.
        """
        if self.embedding_type == "elmo":
            with torch.set_grad_enabled(self.elmo_model.training):
                return self.elmo_model.get_sequence_embeddings(x)  # [B, T, embedding_dim]
        
        else:  # SVD, Skip-gram, CBOW
            return self.embedding_matrix[x]  # [B, T, embedding_dim]