import sys
import torch
from collections import Counter
import torch.nn as nn
import json

from wordsim_score import evaluate_model
from brown import preprocess_brown

import warnings
warnings.filterwarnings("ignore")

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_vocab(sorted_vocab):
    word2idx = {word: i for i, (word, _) in enumerate(sorted_vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

if __name__ == "__main__":
    
    # Get embedding file path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python wordsim.py <embeddings_path>")
        sys.exit(1)
    
    embeddings_path = sys.argv[1]
    
    # Load embeddings
    try:
        embeddings = torch.load(embeddings_path, map_location=DEVICE)
        print(f"Loaded embeddings from {embeddings_path}")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        sys.exit(1)

    if embeddings_path.split('/')[-1].split('.')[0] == 'svd_embeddings': 
        vocab_path = "./archive/sorted_vocab_svd.json"
    else:
        vocab_path = "./archive/sorted_vocab.json" 
        
    with open(vocab_path, "r") as f:
        sorted_vocab = json.load(f)    
    word2idx, idx2word = build_vocab(sorted_vocab)

    # Evaluate the model using WordSim-353
    score = evaluate_model(embeddings, word2idx, DEVICE, generate_plot=True, save_result=True)