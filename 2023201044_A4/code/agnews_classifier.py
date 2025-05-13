import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AGNewsClassifier(nn.Module):
    def __init__(self, embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=False):
        """
        AG News classifier using GRU with embeddings from different sources.

        Args:
            embedding_extractor (EmbeddingExtractor): Extractor handling embeddings.
            rnn_hidden_size (int): GRU hidden size.
            num_classes (int): Number of target classes (default=4).
            fine_tune_elmo (bool): Whether to fine-tune ELMo (ignored for other embeddings).
        """
        super(AGNewsClassifier, self).__init__()
        self.embedding_extractor = embedding_extractor
        self.embedding_type = embedding_extractor.embedding_type

        # Handle ELMo-specific freezing
        if self.embedding_type == "elmo":
            self.embedding_extractor.elmo_model.freeze_initial()
            if not fine_tune_elmo:
                self.embedding_extractor.elmo_model.freeze_gamma()

        # Dynamically determine embedding size:
        # Get the device from the embedding extractor.
        if self.embedding_type == "elmo":
            device0 = next(self.embedding_extractor.elmo_model.parameters()).device
        else:
            device0 = next(self.embedding_extractor.embedding.parameters()).device

        sample_input = torch.randint(0, 10, (1, 5), device=device0)  # Dummy input on the correct device
        embedding_dim = self.embedding_extractor.get_embeddings(sample_input).shape[-1]

        # GRU layer
        self.rnn = nn.GRU(
            input_size=embedding_dim,  # Dynamically determined
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected classifier
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, lengths):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input token indices.
            lengths (torch.Tensor): Sequence lengths.

        Returns:
            torch.Tensor: Logits.
        """
        embeddings = self.embedding_extractor.get_embeddings(x)

        packed_embedded = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed_embedded)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        rnn_out = self.dropout(rnn_out)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Last forward & backward states

        logits = self.classifier(last_hidden)
        return logits