import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import pandas as pd
import numpy as np
import json
import re
import argparse

from elmo_class import ELMo
from agnews_dataset import prepare_agnews_dataloaders
from agnews_classifier import AGNewsClassifier
from embedding_extractor import EmbeddingExtractor
from evaluate import evaluate_model

from load_utils import load_classifier, extend_embeddings
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier(embedding_type, model, train_loader, val_loader, epochs=5, lr=0.001, early_stopping_patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")  # Track best validation loss
    patience_counter = 0  # Track early stopping patience

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        
        for texts, lengths, labels in train_loader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, lengths, labels in val_loader:
                texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)

                outputs = model(texts, lengths)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"../models/best_classifier_{embedding_type}.pth")
            patience_counter = 0  # Reset early stopping counter
        else:
            patience_counter += 1

        # Early stopping check
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    print("Training complete!")


def train_dynamic(train_loader, val_loader, model_path, epochs=5, lambda_mode='frozen'):
    """
    Loads a pre-trained ELMo model (using one of three lambda modes), wraps it in an EmbeddingExtractor,
    instantiates the AGNewsClassifier, trains the classifier on the provided train_loader,
    and returns the trained classifier.

    Args:
        train_loader: DataLoader for the training data.
        model_path: Path to the saved pre-trained ELMo model.
        epochs (int): Number of training epochs for the classifier.
        lambda_mode (str): One of 'trainable', 'frozen', or 'function'.
    Returns:
        classifier: Trained AGNewsClassifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_matrix = torch.load("../archive/elmo_embedding_matrix.pt", map_location=torch.device(device))

    if lambda_mode == 'trainable':
        elmo_model = ELMo(embedding_matrix, lambda_mode="trainable").to(device)
        elmo_model.load_state_dict(torch.load(model_path, map_location=device))
        elmo_model.eval()  # Set to eval mode during embedding extraction
        embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
        classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=True).to(device)
        # print("Lambda parameters (trainable):", elmo_model.gamma.data.cpu().numpy())

    elif lambda_mode == 'frozen':
        elmo_model = ELMo(embedding_matrix, lambda_mode="frozen").to(device)
        elmo_model.load_state_dict(torch.load(model_path, map_location=device))
        elmo_model.eval()  # ELMo remains frozen
        embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
        classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=False).to(device)
        # print("Lambda parameters (frozen):", elmo_model.gamma.cpu().numpy())

    elif lambda_mode == 'function':
        # Load the state dict and modify it to fit the "function" mode expectations.
        state_dict = torch.load(model_path, map_location=device)
        if "gamma" in state_dict:
            del state_dict["gamma"]
        
        elmo_model = ELMo(embedding_matrix, lambda_mode="function").to(device)
        # Initialize function mode parameters using the model's own values
        state_dict["gamma_mlp.0.weight"] = elmo_model.gamma_mlp[0].weight.clone()
        state_dict["gamma_mlp.0.bias"] = elmo_model.gamma_mlp[0].bias.clone()
        elmo_model.load_state_dict(state_dict, strict=False)
        elmo_model.eval()
        embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
        classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=True).to(device)
    else:
        raise ValueError("Invalid lambda_mode. Choose from 'trainable', 'frozen', or 'function'.")
    
    # Train the classifier using your existing training function
    train_classifier("elmo", classifier, train_loader, val_loader, epochs=epochs, lr=0.001)

    # # After training, print the lambda parameters for comparison.
    # if classifier.embedding_extractor.embedding_type == 'elmo':
    #     if lambda_mode in ['trainable', 'frozen']:
    #         # gamma is stored in elmo_model.gamma
    #         print("Lambda parameters ({} mode): {}".format(lambda_mode, elmo_model.gamma.data.cpu().numpy()))
    #     elif lambda_mode == 'function':
    #         # For function mode, get the output of gamma_mlp with a dummy input.
    #         dummy_input = torch.ones(3, device=device)
    #         gamma_weights = elmo_model.gamma_mlp(dummy_input)
    #         print("Lambda parameters (function mode): {}".format(gamma_weights.data.cpu().numpy()))
    
    return classifier


def train_static(embed_path, embedding_type, train_loader, val_loader, epochs=5): 
    embedding_matrix = torch.load(embed_path, map_location=torch.device("cpu"))
    extended_embedding_matrix = extend_embeddings(embedding_matrix).to(device)
    
    # Create an embedding extractor for SVD embeddings.
    embedding_extractor = EmbeddingExtractor(embedding_type=embedding_type, embedding_matrix=extended_embedding_matrix)
    
    # Initialize the downstream classifier using the extractor.
    classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4).to(device)
    train_classifier(embedding_type, classifier, train_loader, val_loader, epochs=epochs, lr=0.001)

    return classifier


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Classifier")
    parser.add_argument("embedding_type", type=str, choices=["elmo", "svd", "cbow", "sgns"],
                        help="Type of embedding to use (elmo, svd, cbow, or sgns)")
    args = parser.parse_args()

    embedding_type = args.embedding_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default model_path(s) for each embedding type
    # You may need to modify these paths according to your setup.
    model_paths = {
        "elmo": "../embeddings/elmo_model_5.pth",
        "svd": "../embeddings/svd_embeddings.pt",
        "cbow": "../embeddings/cbow_embeddings.pt",
        "sgns": "../embeddings/sgns_embeddings.pt"
    }

    with open("../archive/word2idx.json", "r") as f:
        word2idx = json.load(f)
    
    ag_train_loader, ag_val_loader, ag_test_loader = prepare_agnews_dataloaders(
        '../dataset/train.csv', 
        '../dataset/test.csv', 
        word2idx
    )
    

    # For ELMo, ask for lambda mode input.
    if embedding_type == "elmo":
        lambda_mode = input("Enter lambda mode (trainable, frozen, or function): ").strip().lower()
        if lambda_mode not in ["trainable", "frozen", "function"]:
            print("Invalid lambda mode. Please choose from 'trainable', 'frozen', or 'function'.")
            sys.exit(1)
            
        # Call the training function for ELMo.
        # model_path = model_paths["elmo"]
        # classifier = train_dynamic(ag_train_loader, ag_val_loader, model_path, epochs=1, lambda_mode=lambda_mode)
        classifier_path = f"../classifier/classifier_elmo_{lambda_mode}.pth"
        classifier = load_classifier(
            embedding_type='elmo', 
            lambda_mode=lambda_mode, 
            embed_path="../archive/elmo_embedding_matrix.pt", 
            classifier_path=classifier_path
        )
        print('Classifier model loaded. ')

        print("\nEvaluating on Train Set:")
        evaluate_model(classifier, ag_train_loader, device)
        print("\nEvaluating on Test Set:")
        evaluate_model(classifier, ag_test_loader, device)
        
    else:
        # For non-ELMo embeddings, no lambda mode is needed.
        # model_path = model_paths[embedding_type]
        # classifier = train_static(model_path, embedding_type, ag_train_loader, ag_val_loader, epochs=1)
        classifier_path = f"../classifier/classifier_{embedding_type}.pth"
        classifier = load_classifier(
            embedding_type=embedding_type, 
            lambda_mode=None, 
            embed_path=model_paths[embedding_type], 
            classifier_path=classifier_path
        )
        print('Classifier model loaded. ')
        
        print("\nEvaluating on Train Set:")
        evaluate_model(classifier, ag_train_loader, device)
        print("\nEvaluating on Test Set:")
        evaluate_model(classifier, ag_test_loader, device)

if __name__ == "__main__":
    main()

