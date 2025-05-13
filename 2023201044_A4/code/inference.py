#!/usr/bin/env python
import argparse
import torch
import json
from nltk.tokenize import word_tokenize

from text_utils import clean_text
from elmo_class import ELMo
from agnews_classifier import AGNewsClassifier
from embedding_extractor import EmbeddingExtractor


# --- Convert input text to tensor indices using word2idx ---
def text_to_tensor(text, word2idx, max_len=100):
    # Clean the text and tokenize
    text = clean_text(text)
    tokens = word_tokenize(text)
    # In our setup, we assume:
    #   UNK index = len(word2idx)-2
    #   PAD index = len(word2idx)-1
    unk_idx = len(word2idx) - 2
    pad_idx = len(word2idx) - 1
    indices = [word2idx.get(token, unk_idx) for token in tokens]
    # Truncate and pad to max_len
    if len(indices) < max_len:
        indices = indices[:max_len] + [pad_idx] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    # Return as a tensor with batch dimension added
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="ELMo Classifier Inference")
    parser.add_argument("model_path", type=str, help="Path to the saved classifier model")
    parser.add_argument("description", type=str, help="News article description for inference")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load word2idx (assumes you saved it as a JSON file)
    with open("../archive/word2idx.json", "r") as f:
        word2idx = json.load(f)

    # Load the embedding matrix used for training (adjust path as needed)
    embedding_matrix = torch.load("../archive/elmo_embedding_matrix.pt", map_location=device)

    lambda_mode = args.model_path.split("_")[-1].split(".")[0]  
    # print(lambda_mode)
    
    elmo_model = ELMo(embedding_matrix, lambda_mode=lambda_mode).to(device)
    elmo_model.eval()  # Ensure it's in eval mode for inference.
    embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
    
    # Instantiate the classifier architecture.
    # Make sure the architecture matches exactly what was used in training.
    classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=False).to(device)
    
    # Load saved state dict into classifier.
    state_dict = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    # Process input description to get tensor and corresponding length.
    input_tensor = text_to_tensor(args.description, word2idx, max_len=100)
    # Use the actual token count (capped at 100) as length.
    tokens = word_tokenize(clean_text(args.description))
    # print(tokens)
    length = torch.tensor([min(len(tokens), 100)], dtype=torch.long).to(device)

    # Run inference.
    with torch.no_grad():
        logits = classifier(input_tensor, length)  # logits shape: [1, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Print the probabilities for each class.
    for i, prob in enumerate(probs, 1):
        print(f"class-{i} {prob*100:.2f}%")

if __name__ == "__main__":
    main()
