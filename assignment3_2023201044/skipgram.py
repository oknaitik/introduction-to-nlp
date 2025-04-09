import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import json
import itertools
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import Counter

from brown import preprocess_brown
from wordsim_score import evaluate_model

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


brown_corpus = preprocess_brown()
print(f"Total sentences in Brown Corpus: {len(brown_corpus)}")

# Build vocabulary
def build_vocab(sorted_vocab):
    # Build vocabulary
    # word_counts = Counter(word for sentence in brown_corpus for word in sentence)
    
    # Sort word_counts by frequency (descending) to keep order consistent
    # sorted_vocab = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    word2idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word

with open("./archive/sorted_vocab.json", "r") as f:
    sorted_vocab = json.load(f)
word2idx, idx2word = build_vocab(sorted_vocab)

vocab_size = len(word2idx)
print('Vocab size', vocab_size)

# Create a unigram distribution for negative sampling (optimized for GPU)
word_freqs = np.array([freq**0.75 for word, freq in sorted_vocab])
word_freqs = torch.tensor(word_freqs / word_freqs.sum(), dtype=torch.float32)  # keep on CPU

# FRESH ATTEMPT - 2 (EXTREMELY ??)

def generate_training_data(corpus, word2idx, context_size, device="cpu"):
    data = []
    for words in corpus:
        for i in range(context_size, len(words) - context_size):
            target_word = word2idx.get(words[i], 0)
            context_words = [word2idx.get(words[i - j - 1], 0) for j in range(context_size)]
            context_words += [word2idx.get(words[i + j + 1], 0) for j in range(context_size)]
            data.append((target_word, context_words))
    return data

class Word2VecDataset(Dataset):
    def __init__(self, data, word_freqs, vocab_size, negative_samples, device="cpu"):
        self.data = data
        self.word_freqs = word_freqs.to(device)  # Move to GPU if needed
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, context_words = self.data[idx]
        context_words = torch.tensor(context_words, dtype=torch.long, device=self.device)

        # Generate negative samples **efficiently**
        neg_samples = torch.multinomial(self.word_freqs, self.negative_samples, replacement=True)
        
        # Format the data as (x, y) pairs
        positive_pairs = torch.stack([context_words, torch.full_like(context_words, target)], dim=1)  # Label = 1
        negative_pairs = torch.stack([neg_samples, torch.full_like(neg_samples, target)], dim=1)  # Label = 0
        
        # Combine positive and negative samples
        samples = torch.cat([positive_pairs, negative_pairs])
        labels = torch.cat([torch.ones(len(context_words)), torch.zeros(self.negative_samples)])

        return samples, labels

def collate_fn(batch):
    samples, labels = zip(*batch)  # Unpack from dataset
    samples = torch.cat(samples)  # Convert list of tensors to batch tensor
    labels = torch.cat(labels)
    return samples, labels

class SkipGramBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device="cpu"):
        super(SkipGramBinaryClassifier, self).__init__()
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)

        nn.init.xavier_uniform_(self.embeddings.weight, gain=0.1)
        nn.init.xavier_uniform_(self.context_embeddings.weight, gain=0.1)

    def forward(self, pairs):
        context, target = pairs[:, 0], pairs[:, 1]  # Extract context and target words
        context_embedding = self.embeddings(context)
        target_embedding = self.context_embeddings(target)
        
        score = torch.sum(context_embedding * target_embedding, dim=1)  # Dot product
        return torch.sigmoid(score)  # Binary classification

def train_word2vec(training_data, config, word_freqs, device="cpu", patience=3):
    dataset = Word2VecDataset(training_data, word_freqs, config['vocab_size'], config['negative_samples'], device=device)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0, 
        pin_memory=(device.type == 'cpu'), 
        collate_fn=collate_fn
    )

    model = SkipGramBinaryClassifier(config['vocab_size'], config['embedding_dim'], device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    best_score = -1
    best_model_state = None
    early_stop_count = 0  # Tracks consecutive epochs without improvement

    for epoch in range(config['epochs']):
        start_time = time.time()
        model.train()
        total_loss = 0

        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate Model on WordSim-353
        model.eval()
        score = evaluate_model(model.embeddings.weight, word2idx, device)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, WordSim Score: {score:.4f}, Time: {epoch_time:.1f}s")

        # Check if the model improved
        if score > best_score:
            best_score = score
            best_model_state = model.state_dict()
            early_stop_count = 0  # Reset early stopping counter
            print("✅ Best model updated!")
        else:
            early_stop_count += 1
            print(f"🚨 No improvement for {early_stop_count}/{patience} epochs.")

        # Early stopping condition
        if early_stop_count >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs with no improvement!")
            break

    # Load best model before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

"""
Hyper-parameter Optimization

# Define parameter choices
param_grid = {
    'batch_size': [512], 
    'negative_samples': [3, 5, 7], 
    'embedding_dim': [300],  
    'context_size': [2, 3]
}

# Generate all possible combinations
param_combinations = list(itertools.product(*param_grid.values()))

best_model = None
best_score = float('-inf')  # Replace with evaluation metric (e.g., similarity score)

# cache training data 
training_data_cached = {context_size: generate_training_data(brown_corpus, word2idx, context_size=context_size, device=device) for context_size in param_grid['context_size']}
print('Training data generated.')

for params in param_combinations:
    config = {
        'vocab_size': len(word2idx),
        'embedding_dim': params[2],
        'batch_size': params[0],
        'epochs': 10,  # Reduce epochs for faster tuning
        'learning_rate': 0.001,
        'negative_samples': params[1],
        'context_size': params[3]
    }
    
    print(f"Training with: {config}")
    
    # Generate training data with the selected context size
    training_data = training_data_cached[config['context_size']]
    
    # Train model
    model = train_word2vec(training_data, config, word_freqs.to(device), device=device)
    
    # Evaluate on a small word similarity task
    score = evaluate_model(model, word2idx, device)  # compare word similarities
    
    # Track the best configuration
    if score > best_score:
        best_score = score
        best_model = model
        best_config = config

print("Best config:", best_config)
"""

# best hyper-params: {'vocab_size': 40654, 'embedding_dim': 300, 'batch_size': 512, 'epochs': 20, 'learning_rate': 0.001, 'negative_samples': 7, 'context_size': 2} 
# Corr: 0.2543 with 10-epoch training 

config = {'vocab_size': 40654, 'embedding_dim': 300, 'batch_size': 512, 'epochs': 1, 'learning_rate': 0.001, 'negative_samples': 7, 'context_size': 2} 

# Generate training data with the selected context size
training_data = generate_training_data(brown_corpus, word2idx, context_size=config['context_size'], device=device)

# Train model
model = train_word2vec(training_data, config, word_freqs.to(device), device=device)

embedding_weights = model.embeddings.weight.detach().cpu()  # Ensure it's on CPU before saving
torch.save(embedding_weights, "sgns_embeddings.pt")