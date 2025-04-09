import pandas as pd
import torch
import torch.nn.functional as F 
from scipy.stats import spearmanr
from plot import generate_plots
from result import save_results

def evaluate_model(embedding_weights, word2idx, device, generate_plot=False, save_result=False, dataset_path="./archive/wordsim353crowd.csv"):
    """Evaluate a trained word2vec model using Spearman’s Rank Correlation on WordSim-353 dataset."""
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Get word embeddings from model
    # embedding_weights = model.embeddings.weight.to(device)
    embedding_weights = embedding_weights.to(device)
    vocab_size, embed_dim = embedding_weights.shape
    
    # Create an embedding lookup module
    word_vectors = torch.nn.Embedding(vocab_size, embed_dim)
    word_vectors.load_state_dict({'weight': embedding_weights})
    word_vectors.to(device)
    word_vectors.eval()  # Set to evaluation mode

    def cosine_similarity(word1, word2):
        """Compute cosine similarity between two words using learned embeddings."""
        if word1 not in word2idx or word2 not in word2idx:
            return None  # Skip missing words
        
        vec1 = word_vectors(torch.tensor(word2idx[word1], device=device)).detach()
        vec2 = word_vectors(torch.tensor(word2idx[word2], device=device)).detach()
        
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    # Compute similarity scores
    model_similarities = []
    human_scores = []
    word_pairs = []

    for _, row in df.iterrows():
        word1, word2, human_score = row["Word 1"], row["Word 2"], row["Human (Mean)"]
        
        sim = cosine_similarity(word1, word2)
        if sim is not None:
            word_pairs.append((word1, word2))
            model_similarities.append(sim)
            human_scores.append(human_score)

    # Compute Spearman’s Rank Correlation
    spearman_corr, _ = spearmanr(model_similarities, human_scores)
    print(f"Spearman’s Rank Correlation: {spearman_corr:.4f}")

    if save_result: 
        save_results(word_pairs, human_scores, model_similarities)
    
    if generate_plot: 
        generate_plots(model_similarities, human_scores, spearman_corr)
        
    return spearman_corr