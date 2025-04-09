from brown import preprocess_brown
from wordsim_score import evaluate_model

import pandas as pd
import numpy as np
from functools import partial
from collections import Counter
import random
import json
import torch
from torch import optim
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import tqdm
import itertools
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading corpus...")
brown_corpus = preprocess_brown()
print(f"Total sentences in Brown Corpus: {len(brown_corpus)}")

# Build vocabulary
def build_vocab(corpus):
    # # Build vocabulary
    # word_counts = Counter(word for sentence in brown_corpus for word in sentence)
    # # Sort word_counts by frequency (descending) to keep order consistent
    # sorted_vocab = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Assign indices, reserving 0 for padding and 1 for unknown words (OOV handling)
    word2idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

with open("./archive/sorted_vocab.json", "r") as f:
    sorted_vocab = json.load(f)
word2idx, idx2word = build_vocab(sorted_vocab)
VOCAB_SIZE = len(word2idx)

word_freqs = np.array([freq**0.75 for word, freq in sorted_vocab])
word_freqs = torch.tensor(word_freqs / word_freqs.sum(), dtype=torch.float32)  # keep on CPU

# Generate CBOW Training Data
def generate_training_data(corpus, word2idx, context_size):
    data = []
    for sentence in corpus:
        if len(sentence) < 2 * context_size + 1:
            continue  # Skip short sentences
        
        for i in range(context_size, len(sentence) - context_size):
            context = [sentence[j] for j in range(i - context_size, i + context_size + 1) if j != i]
            target = sentence[i]
            data.append((context, target))
    
    return data

class CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWNegativeSampling, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize weights
        init_range = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.output_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, context_words, target_words, negative_samples):
        # Compute word embeddings
        context_vec = self.embeddings(context_words).mean(dim=1)  # Avg context vectors
        target_vec = self.output_embeddings(target_words)  # Target word vectors
        neg_vec = self.output_embeddings(negative_samples)  # Negative samples

        # Compute positive and negative scores
        pos_score = torch.bmm(target_vec.unsqueeze(1), context_vec.unsqueeze(2)).squeeze()
        neg_score = torch.bmm(neg_vec, context_vec.unsqueeze(2)).squeeze()

        # Compute loss
        loss = - torch.log(torch.sigmoid(pos_score)) - torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)
        return loss.mean()

def get_negative_samples(batch_size, word_freqs, neg_samples):
    """
    Samples negative words based on the unigram distribution raised to the 3/4 power.

    Args:
        batch_size (int): Number of samples in the batch.
        word_freqs (Tensor): Precomputed unigram distribution (freq**0.75 / sum(freq**0.75)).
        neg_samples (int): Number of negative samples per target word.

    Returns:
        Tensor: A tensor of shape (batch_size, neg_samples) containing sampled negative word indices.
    """
    return torch.multinomial(word_freqs, batch_size * neg_samples, replacement=True).view(batch_size, neg_samples).to(DEVICE)


"""
Hyper-parameter Optimization.

# Define hyperparameter grid
CONTEXT_SIZES = [2, 3, 5]  # Context window sizes
NEGATIVE_SAMPLES_LIST = [2, 5, 7]  # Number of negative samples
EMBEDDING_DIMS = [300]  # Embedding dimensions
PATIENCE = 3 # stop training if no improvement

# Store the best configuration
best_score = -1
best_config = None
best_model_state = None

# Iterate over all combinations of hyperparameters
for context_size, neg_samples, embedding_dim in itertools.product(CONTEXT_SIZES, NEGATIVE_SAMPLES_LIST, EMBEDDING_DIMS):
    print(f"🔍 Training with CONTEXT_SIZE={context_size}, NEGATIVE_SAMPLES={neg_samples}, EMBEDDING_DIM={embedding_dim}")

    # Prepare training data with the new context size
    def get_batches(data, batch_size):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            contexts = torch.tensor([[word2idx.get(word, 0) for word in context] for context, _ in batch], dtype=torch.long, device=DEVICE)
            targets = torch.tensor([word2idx.get(target, 0) for _, target in batch], dtype=torch.long, device=DEVICE)
            negatives = get_negative_samples(len(batch), VOCAB_SIZE, neg_samples)
            yield contexts, targets, negatives

    # Initialize Model with current hyperparams
    model = CBOWNegativeSampling(VOCAB_SIZE, embedding_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stop_count = 0  # Tracks no-improvement epochs
    best_epoch_score = -1  # Best score for this model
    
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for contexts, targets, negatives in get_batches(training_data, BATCH_SIZE):
            optimizer.zero_grad()
            loss = model(contexts, targets, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate model on WordSim-353
        model.eval()
        score = evaluate_model(model.embeddings.weight, word2idx, DEVICE)

        # Print results
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | WordSim Score: {score:.4f}")

        # Track the best model for this configuration
        if score > best_epoch_score:
            best_epoch_score = score
            early_stop_count = 0  # Reset early stopping count
        else:
            early_stop_count += 1  # Increment if no improvement

        # Early stopping check
        if early_stop_count >= PATIENCE:
            print(f"🚨 Early stopping triggered after {PATIENCE} epochs of no improvement.")
            break  # Stop training early

    # Track overall best configuration
    if best_epoch_score > best_score:
        best_score = best_epoch_score
        best_config = (context_size, neg_samples, embedding_dim)
        best_model_state = model.state_dict()
        
    print(f"✅ Finished training for CONTEXT_SIZE={context_size}, NEGATIVE_SAMPLES={neg_samples}, EMBEDDING_DIM={embedding_dim}\n")

# Save the best model
torch.save(best_model_state, f"/kaggle/working/best_cbow_model_context_size_{best_config[0]}_neg_samples_{best_config[1]}_embed_dim_{best_config[2]}.pt")
print(f"🏆 Best Config: CONTEXT_SIZE={best_config[0]}, NEGATIVE_SAMPLES={best_config[1]}, EMBEDDING_DIM={best_config[2]}")
print(f"🔥 Best WordSim Score: {best_score:.4f}")
"""


"""
Model training. Load and Use pre-trained model, if available.
"""
# Best hyperparams after optimization (WordSim Score: 0.222)
CONTEXT_SIZE = 2  # Number of words on the left and right of the target word
NEGATIVE_SAMPLES = 7  # Number of negative samples per word
EMBEDDING_DIM = 300
BATCH_SIZE = 512
EPOCHS = 1
LEARNING_RATE = 0.001

training_data = generate_training_data(brown_corpus, word2idx, CONTEXT_SIZE)

# # Prepare training data
# def get_batches(data, batch_size):
#     random.shuffle(data)
#     for i in range(0, len(data), batch_size):
#         batch = data[i : i + batch_size]
#         contexts = torch.tensor([[word2idx.get(word, 0) for word in context] for context, _ in batch], dtype=torch.long, device=DEVICE)
#         targets = torch.tensor([word2idx.get(target, 0) for _, target in batch], dtype=torch.long, device=DEVICE)
#         negatives = get_negative_samples(len(batch), VOCAB_SIZE, NEGATIVE_SAMPLES)
#         yield contexts, targets, negatives

def get_batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        contexts = torch.tensor([[word2idx.get(word, 0) for word in context] for context, _ in batch], dtype=torch.long, device=DEVICE)
        targets = torch.tensor([word2idx.get(target, 0) for _, target in batch], dtype=torch.long, device=DEVICE)
        negatives = get_negative_samples(len(batch), word_freqs, NEGATIVE_SAMPLES)
        yield contexts, targets, negatives

# Initialize Model
model = CBOWNegativeSampling(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_score = -1  # Track best WordSim-353 score
early_stop_count = 0  # Early stopping counter
PATIENCE = 3  # Stop if no improvement after 3 epochs

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    # Training Phase
    for contexts, targets, negatives in get_batches(training_data, BATCH_SIZE):
        optimizer.zero_grad()
        loss = model(contexts, targets, negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate using WordSim-353
    model.eval()
    score = evaluate_model(model.embeddings.weight, word2idx, DEVICE)

    # Print Epoch Summary
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | WordSim Score: {score:.4f}")

    # Save Best Model
    if score > best_score:
        best_score = score
        # torch.save(model.state_dict(), "/kaggle/working/best_cbow_model.pt")
        # print("Best model saved!")
        torch.save(model.embeddings.weight.data, 'cbow_embeddings.pt')
        print("Best model embeddings saved.")

        early_stop_count = 0  # Reset counter since improvement happened
    else:
        early_stop_count += 1  # No improvement, increment counter
        if early_stop_count >= PATIENCE:
            print(f"🚨 Early stopping triggered! No improvement for {PATIENCE} epochs.")
            break  # Stop training early

