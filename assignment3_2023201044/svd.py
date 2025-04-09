import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from nltk.corpus import stopwords
from wordsim_score import evaluate_model
from brown import preprocess_brown

import warnings
warnings.filterwarnings("ignore")

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define stopwords
STOPWORDS = set(stopwords.words("english"))


brown_corpus = preprocess_brown()
print(f"Total sentences in Brown Corpus: {len(brown_corpus)}")

def build_vocab(corpus, MIN_WORD_FREQUENCY = 3):
    SPECIAL_TOKENS = ["<unk>"]  # You can add more special tokens if needed
    
    # Step 1: Count word frequencies
    word_counts = Counter(word for sentence in corpus for word in sentence)
    
    # Step 2: Filter words by min frequency and create vocabulary
    vocab_words = SPECIAL_TOKENS + [word for word, count in word_counts.items() if count >= MIN_WORD_FREQUENCY]
    vocab_words.sort()
    
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}

    # save sorted vocab for wordsim task
    sorted_vocab = list()
    for word in vocab_words:
        sorted_vocab.append((word, word_counts[word]))
    
    with open("sorted_vocab.json", "w") as f:
        json.dump(sorted_vocab, f)
        
    return word_counts, word2idx

MIN_WORD_FREQUENCY = 3
word_counts, word2idx = build_vocab(brown_corpus, MIN_WORD_FREQUENCY)

# Define function to build the co-occurrence matrix
def build_co_occurrence_matrix(sentences, window_length, apply_weighting, remove_stopwords):
    V = len(word2idx)
    co_occurrence_matrix = np.zeros((V, V))
    
    for sent in sentences:
        for idx, word in enumerate(sent):
            if remove_stopwords and word in STOPWORDS:
                continue  

            for context_id in range(max(0, idx - window_length), min(len(sent), idx + window_length + 1)):  
                context_word = sent[context_id]
                
                if remove_stopwords and context_word in STOPWORDS:
                    continue

                row = word2idx[word if word_counts[word] >= MIN_WORD_FREQUENCY else '<unk>']
                col = word2idx[context_word if word_counts[context_word] >= MIN_WORD_FREQUENCY else '<unk>']
                
                # Apply weighting based on distance
                weight = 1 / (abs(idx - context_id) + 1) if apply_weighting else 1  
                
                co_occurrence_matrix[row][col] += weight  

    return co_occurrence_matrix
    
"""
Hyper-parameter Optimization.

# Window sizes and embedding dimensions to test
WINDOW_SIZES = [2, 4, 6]
EMBEDDING_DIMS = [100, 200, 300]

# Store results
results = []

# Iterate over all combinations
for apply_weighting in [False, True]:  # Without weighting, with weighting
    for remove_stopwords in [False, True]:   # With stopwords, without stopwords
        for window_size in WINDOW_SIZES:
            # Build co-occurrence matrix for this setting
            co_occurrence_matrix = build_co_occurrence_matrix(brown_corpus, window_size, apply_weighting, remove_stopwords)
            
            # Compute SVD
            U, S, Vt = svds(co_occurrence_matrix, k=max(EMBEDDING_DIMS))
            U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]  # Reverse order
            
            for embedding_dim in EMBEDDING_DIMS:

                # Extract and normalize word vectors
                word_vectors = U[:, : embedding_dim] * S[: embedding_dim]
                print(word_vectors.shape)
                
                norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid NaNs
                word_vectors /= norms

                # save the word embeddings 
                embeddings_path = "/kaggle/working/svd_embeddings.pt"
                torch.save(torch.tensor(word_vectors, dtype=torch.float32), embeddings_path)

                # Evaluate model
                embeddings = torch.load(embeddings_path, map_location=DEVICE)
                correlation = evaluate_model(embeddings, word2idx, DEVICE)

                # Store results
                results.append({
                    "Weighting": apply_weighting,
                    "Remove Stopwords": remove_stopwords,
                    "Window Size": window_size,
                    "Embedding Dim": embedding_dim,
                    "Spearman Correlation": correlation
                })

                print(f"Weighting: {apply_weighting}, Remove Stopwords: {remove_stopwords}, Window Size: {window_size}, Embedding Dim: {embedding_dim} → Spearman Correlation: {correlation:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV (optional)
results_df.to_csv("evaluation_results.csv", index=False)

# Display final results
print(results_df)

"""
# best hyperparams
apply_weighting, remove_stopwords, window_size, embedding_dim = False, True, 4, 100

# Build co-occurrence matrix for this setting
co_occurrence_matrix = build_co_occurrence_matrix(brown_corpus, window_size, apply_weighting, remove_stopwords)

# Compute SVD
U, S, Vt = svds(co_occurrence_matrix, k=embedding_dim)
U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]  # Reverse order

# Extract and normalize word vectors
word_vectors = U[:, : embedding_dim] * S[: embedding_dim]
print("Embeddings dimensions:", word_vectors.shape)

norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid NaNs
word_vectors /= norms

# save the word embeddings 
embeddings_path = "./svd_embeddings.pt"
torch.save(torch.tensor(word_vectors, dtype=torch.float32), embeddings_path)

# Evaluate model
embeddings = torch.load(embeddings_path, map_location=DEVICE)
correlation = evaluate_model(embeddings, word2idx, DEVICE)

  