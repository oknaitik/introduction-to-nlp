import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import sys
import optuna
import os
import re
from functools import partial
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import itertools
import pandas as pd

from tokenizer import clean_and_tokenize_1, clean_and_tokenize_2

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Load Glove embeddings
def load_glove(file_path):
    glove_dict = {}
    with open(file_path, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = [float(x) for x in values[1:]]
            glove_dict[word] = word_embedding
    return glove_dict

# Text preprocessing
def clean_and_tokenize_1(text):
    # Merge broken lines (preserve paragraph breaks)
    text = re.sub(r"(?<!\.)\n(?!\n)", " ", text)  
    
    # Remove specified sections
    text = re.sub(r"The Project Gutenberg.*?CHAPTER I\.", "", text, flags=re.DOTALL)
    text = re.sub(r"END OF VOL\. I\..*?CHAPTER I\.", "", text, flags=re.DOTALL)
    text = re.sub(r"END OF THE SECOND VOLUME\..*?CHAPTER I\.", "", text, flags=re.DOTALL)
    text = re.sub(r"Transcriber[’']s note.*", "", text, flags=re.DOTALL)
    text = re.sub(r"CHAPTER [IVXLCDM]+\.?", "", text)

    # Lowercase all text
    text = text.lower()

    # Replace patterns with placeholders
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)  # URLs
    text = re.sub(r"#\w+", "<HASHTAG>", text)  # Hashtags
    text = re.sub(r"@\w+", "<MENTION>", text)  # Mentions
    text = re.sub(r"\b(?:mr|mrs|ms)\b\.?", "<TITLE>", text)  # Titles
    text = re.sub(r"\b\w+@\w+\.\w+\b", "<EMAIL>", text)  # Emails
    text = re.sub(r"\b\d+\b", "<NUM>", text) # Numbers

    # Remove underscores around text
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Replace ! or ? with ". "
    text = re.sub(r"[!?]", ". ", text)

    # Replace all other punctuations with space
    text = re.sub(r"[^\w\s.]", " ", text)

    # Expand contractions using regex-based approach
    text = re.sub(r"([a-z]+)n[\'’]t", r"\1 not", text)  # don't → do not, isn't → is not
    text = re.sub(r"([i])[\'’]m", r"\1 am", text)  # I'm → I am
    text = re.sub(r"([a-z]+)[\'’]s", r"\1 is", text)  # it's → it is, he's → he is

    sentences = re.split(r'(?<=[.!?])(?:["”]?)\s+', text)  # End only on ., !, ?, .", !", ?"
    # return "\n".join(sentences)
    sentences = [s.strip() for s in sentences if s.strip()]  
    
    tokenized_sentences = []
    # small = 0
    for sentence in sentences:
        # **Tokenization (preserve punctuation as separate tokens)**
        tokens = re.findall(r'\w+|[.,!?"]', sentence)
        # if len(tokens) < 6: # drop sentences that are very small
        #     continue
            
        if tokens[-1] == '.':
            tokens.pop()
        tokenized_sentence = ["SOS"] + tokens + ["EOS"]
        tokenized_sentences.append(" ".join(tokenized_sentence))
        # tokenized_sentences.append(" ".join(tokens))

    # print(small)
    return "\n".join(tokenized_sentences)

def clean_and_tokenize_2(text):
    # Merge broken lines (preserve paragraph breaks)
    text = re.sub(r"(?<!\.)\n(?!\n)", " ", text)  
    
    # Remove specified sections
    # Remove everything from and after "[ 18 ]"
    text = re.sub(r"Brightdayler.*", "", text, flags=re.DOTALL)

    # Lowercase all text
    text = text.lower()

    # Remove underscores around text
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Replace patterns with placeholders
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)  # URLs
    text = re.sub(r"#\w+", "<HASHTAG>", text)  # Hashtags
    text = re.sub(r"@\w+", "<MENTION>", text)  # Mentions
    text = re.sub(r"\b(?:mr|mrs|ms)\b\.?", "<TITLE>", text)  # Titles
    text = re.sub(r"\b\w+@\w+\.\w+\b", "<EMAIL>", text)  # Emails
    text = re.sub(r"\b\d+\b", "<NUM>", text) # Numbers

    # Expand contractions using regex-based approach
    text = re.sub(r"([a-z]+)n[\'’]t", r"\1 not", text)  # don't → do not, isn't → is not
    text = re.sub(r"([i])[\'’]m", r"\1 am", text)  # I'm → I am
    text = re.sub(r"([a-z]+)[\'’]s", r"\1 is", text)  # it's → it is, he's → he is

    # Replace ! or ? with ". "
    # text = re.sub(r"[!?]", ". ", text)

    # The corpus contains too many !, ?. Therefore we use only period(.) to reduce smaller sentences
    # Replace all other punctuations with space
    text = re.sub(r"[^\w\s.]", " ", text)

    # In the corpus, there are no quoted lines! So, we don't need "" as sentence separator
    # sentences = re.split(r'(?<=[.!?])(?:["”]?)\s+', text)  # End only on ., !, ?, .", !", ?"
    sentences = re.split(r'(?<=\.)\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]  
    
    tokenized_sentences = []
    for sentence in sentences:
        # **Tokenization (preserve punctuation as separate tokens)**
        tokens = re.findall(r'\w+|[.,!?"]', sentence)
        # if len(tokens) < 6: # drop sentences that are very small
        #     continue
            
        if tokens[-1] == '.':
            tokens.pop()
        tokenized_sentence = ["SOS"] + tokens + ["EOS"]
        tokenized_sentences.append(" ".join(tokenized_sentence))

    # print(small)
    return "\n".join(tokenized_sentences[3: ])

# Build vocab
def build_vocab(sentences, glove_dict): 
    vocab = {'<UNK>': 0}
    for sent in sentences:
        for word in sent:
            if word in glove_dict and word not in vocab:
                vocab[word] = len(vocab)
    return vocab

# Build n-grams
def build_ngrams(sentences, vocab, n=5):
    inputs, labels = [], []
    for sent in sentences:
        if len(sent) < n:
            continue

        if sent[0] == 'SOS': 
            sent = ['SOS'] * (n-2) + sent # pre-pend (n-2) SOS tokens 
            
        for i in range(len(sent) - (n-1)):
            prefix = sent[i: i + n-1]
            label = sent[i + n-1]
            encoded_prefix = [vocab.get(word, vocab['<UNK>']) for word in prefix]
            encoded_label = vocab.get(label, vocab['<UNK>'])
            inputs.append(encoded_prefix)
            labels.append(encoded_label)
    return inputs, labels

def build_embedding_matrix(vocab, glove_dict, embedding_dim=300):
    embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)  # Initialize matrix

    for i, word in enumerate(vocab):
        embedding_vector = glove_dict.get(word)  # Fetch from GloVe
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype=np.float32)  # Assign NumPy array

    return embedding_matrix
    
# FFNN model
class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, embedding_matrix, n=5, dropout=0.5):
        super(FFNN, self).__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False).to(device)
        self.fc1 = nn.Linear(embedding_dim * (n-1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.embeddings(x)
        x = x.view(x.size(0), -1)
        x = self.gelu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Train function
def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, early_stop_patience=3):
    best_val_loss = float('inf')
    count_early_stopping = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, loss_fn, val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            count_early_stopping = 0
        else:
            count_early_stopping += 1
            if count_early_stopping == early_stop_patience:
                print("Early stopping triggered.")
                break

# Evaluation function
def evaluate(model, loss_fn, data_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
    return val_loss/ len(data_loader)

# Perplexity calculation
def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            # loss.item is the average loss per token in the batch
            # to get the total loss for that batch, multiply the number of tokens in the batch
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def calculate_perplexities_and_save(model, data_loader, sentences, vocab, file_path):
    """Calculate average perplexity and sentence-wise perplexities, then save to a file."""
    model.eval()
    total_loss = 0
    sentence_perplexities = []
    # print(len(data_loader), len(data_loader.dataset), len(sentences))

    with torch.no_grad():
        for inputs, labels, sentence in zip(data_loader.dataset.tensors[0], data_loader.dataset.tensors[1], sentences):
            inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
            labels = labels.unsqueeze(0).to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            perplexity = torch.exp(loss).item()
            
            sentence_perplexities.append((' '.join(sentence), perplexity))
            total_loss += loss.item() * inputs.size(0)

    # Compute average perplexity
    avg_loss = total_loss/ len(sentences)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Write results to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{avg_perplexity:.2f}\n")  # Write average perplexity
        for sentence, perplexity in sentence_perplexities:
            f.write(f"{sentence}\t{perplexity:.2f}\n")  # Write each sentence's perplexity

    print(f"Perplexities saved to {file_path}")
    return avg_perplexity

# Optuna Hyperparameter Optimization
def optimize_hyperparams(train_dataset, val_dataset, test_dataset, vocab, glove_dict, n=5, num_trials=20, num_epochs=5):
    def objective(trial):
        # Sample hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.5])
        optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        embedding_dim = 300

        # Create DataLoaders with the sampled batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize the model
        embedding_matrix = build_embedding_matrix(vocab, glove_dict, embedding_dim)
        
        model = FFNN(embedding_dim, hidden_dim, len(vocab), embedding_matrix, n=n, dropout=dropout_rate).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Choose optimizer
        if optimizer_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Train the model
        train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs)

        # Load best model and evaluate on validation set
        model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
        val_perplexity = calculate_perplexity(model, val_loader)

        return val_perplexity  # Optuna minimizes perplexity

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    # Return best hyperparameters
    return study.best_params

# Prediction function
def predict_top_k_words(model, input_sentence, vocab, index_to_word, k=5, n=5):
    model.eval()
    input_sentence = input_sentence.lower()
    input_tokens = input_sentence.split()
    
    if len(input_tokens) < n-1:
        print(f"Sentence should have at least {n-1} words for an {n}-gram model.")
        return []

    input_tokens = ['SOS'] + input_tokens
    
    context_words = input_tokens[-(n-1):]
    encoded_input = [vocab.get(word, vocab['<UNK>']) for word in context_words]
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)

    # word_to_find = "headache"
    word_to_find = 'widow'

    # Ensure the word exists in the vocabulary, otherwise use <UNK>
    word_index = vocab.get(word_to_find, vocab["<UNK>"])
    
    with torch.no_grad():
        output = model(input_tensor)
        output_probs = torch.softmax(output, dim=-1).squeeze(0)  # Remove batch dim
        
        unk_index = vocab["<UNK>"]
        
        # Compute total probability excluding <UNK>
        valid_probs = output_probs.clone()
        valid_probs[unk_index] = 0  # Zero out <UNK> probability
        total_valid_prob = valid_probs.sum().item()  # New normalization factor

        # Get top-k predictions
        top_k = torch.topk(output_probs, k + 1)  # Fetch extra in case <UNK> appears
        top_k_indices = top_k.indices.tolist()
        top_k_probs = top_k.values.tolist()

        # Filter out <UNK> and normalize
        filtered_predictions = [
            (index_to_word[idx], prob / total_valid_prob)  # Normalize with full valid sum
            for idx, prob in zip(top_k_indices, top_k_probs) if idx != unk_index
        ]

        # Probability of the specific word
        word_prob = (output_probs[word_index].item() / total_valid_prob) if word_index != unk_index else 0.0

    print(f"Proba('{word_to_find}'): {word_prob:.6f}")

    return filtered_predictions[:k]

def build_vocab_seq(sentences, glove_dict):
    vocab = {'<UNK>': 0 ,'<PAD>':1}
    for sent in sentences:
        for word in sent:
            if word in glove_dict and word not in vocab:
                vocab[word] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, sentences, vocab, embedding_matrix):
        self.sentences = sentences
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix

        # Filter out empty sequences and sequences with zero length
        self.data = self.create_sequences()

    def create_sequences(self):
        sequences = []
        for sent in self.sentences:
            if len(sent) > 1:  # Ensure the sentence has more than one word
                input_seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in sent[:-1]]
                output_seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in sent[1:]]
                if len(input_seq) > 0 and len(output_seq) > 0:  # Check if sequences are non-empty
                    sequences.append((input_seq, output_seq))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]
        input_embeds = torch.tensor(input_seq, dtype=torch.long)
        output_seq = torch.tensor(output_seq, dtype=torch.long)
        return input_embeds, output_seq

def get_collate_fn(vocab):
    return partial(collate_fn_seq, vocab=vocab)
    
def collate_fn_seq(batch, vocab):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=vocab['<PAD>'])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=vocab['<PAD>'])
    lengths = [len(seq) for seq in sequences]
    return sequences_padded, targets_padded,torch.tensor(lengths)

class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers=2, dropout=0.5):
        super(RNNModel, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                          dropout=dropout, batch_first=True, nonlinearity='tanh')

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_seq, lengths, hidden_state=None):
        # Embed the input sequence
        embedded = self.dropout(self.embedding(input_seq))

        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(input_seq.device)
            hidden_state = h_0

        packed_output, hidden_state = self.rnn(packed_input, hidden_state)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        output = self.fc(rnn_out)
        return output, hidden_state

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_seq, lengths, hidden_state=None):
        # Embed the input sequence
        embedded = self.dropout(self.embedding(input_seq))

        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        if hidden_state is None:
            h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(input_seq.device)
            c_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(input_seq.device)
            hidden_state = (h_0, c_0)

        packed_output, hidden_state = self.lstm(packed_input, hidden_state)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(lstm_out)
        return output, hidden_state

    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h_0, c_0

def train_seq(model, optimizer, loss_fn, vocab_size, train_loader, val_loader, num_epochs=10, early_stop_patience=3):
    # vocab_size = len(vocab)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_seq, target_seq, lengths in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Initialize hidden state at the start of each batch
            hidden = model.init_hidden(input_seq.size(0), device)
            optimizer.zero_grad()

            # Forward pass
            output, hidden = model(input_seq, lengths, hidden)

            # Reshape output and target for the loss calculation
            loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss/ len(train_loader): .3f}", end=" ")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq, lengths in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                # No need to pass hidden state between batches during evaluation
                output, _ = model(input_seq, lengths)

                # Calculate validation loss
                loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1))
                val_loss += loss.item()

        # Calculate average validation loss
        print(f"Validation Loss: {val_loss/ len(val_loader):.3f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), '/kaggle/working/best_model2.pth')
            # print(f"Model saved with validation loss: {best_val_loss:.3f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping!")
                break

def calculate_perplexity_seq(model, data_loader, vocab):
    """
    Calculate the perplexity of the model on the provided data_loader.

    Args:
        model (nn.Module): The trained LSTM model.
        data_loader (DataLoader): DataLoader for the dataset (test or validation).
        vocab (dict): Vocabulary dictionary mapping words to indices.
        device (torch.device): The device to run the computations on.

    Returns:
        float: The calculated perplexity.
    """
    model.eval()
    data_loss = 0
    vocab_size=len(vocab)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

    with torch.no_grad():
        for input_seq, target_seq, lengths in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # No need to pass hidden state between batches during evaluation
            output, _ = model(input_seq, lengths)

            # Calculate validation loss
            loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1))
            data_loss += loss.item()

    # Calculate average validation loss
    data_loss /= len(data_loader)
    perplexity=torch.exp(torch.tensor(data_loss))
    return perplexity

def calculate_perplexities_and_save_seq(model, data_loader, sentences, vocab, file_path):
    model.eval()
    total_loss = 0
    sentence_perplexities = []
    vocab_size = len(vocab)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])  # Ignore padding tokens
    
    # print(f"Total batches: {len(data_loader)}, Total sentences: {len(sentences)}")
    
    with torch.no_grad():
        for (input_seq, target_seq), sentence in zip(data_loader.dataset, sentences):
            input_seq = input_seq.unsqueeze(0).to(device)  # Add batch dimension
            target_seq = target_seq.unsqueeze(0).to(device)

            # Compute length dynamically
            lengths = torch.tensor([input_seq.shape[1]], dtype=torch.int64).cpu()

            # Forward pass through the model
            outputs, _ = model(input_seq, lengths)

            # Compute loss
            loss = loss_fn(outputs.view(-1, vocab_size), target_seq.view(-1))
            perplexity = torch.exp(loss).item()

            # Store sentence-wise perplexity
            sentence_perplexities.append((' '.join(sentence), perplexity))
            total_loss += loss.item()

    # Compute average perplexity
    avg_loss = total_loss / len(sentences)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Write results to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{avg_perplexity:.2f}\n")  # Write average perplexity
        for sentence, perplexity in sentence_perplexities:
            f.write(f"{sentence}\t{perplexity:.2f}\n")  # Write each sentence's perplexity

    print(f"Perplexities saved to {file_path}")
    return avg_perplexity

# Optuna Hyperparameter Optimization
def optimize_hyperparams_seq(lm_type, train_dataset, val_dataset, test_dataset, vocab, glove_dict, num_trials=20, num_epochs=5):
    def objective(trial):
        # Sample hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.5])
        optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        embedding_dim = 300

        # Use partial to pass vocab when calling DataLoader
        collate_fn_seq = get_collate_fn(vocab)
        
        # Create DataLoaders with the sampled batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_seq)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_seq)

        # Initialize the model
        embedding_matrix = build_embedding_matrix(vocab, glove_dict, embedding_dim)
        if lm_type == 'r':
            model = RNNModel(embedding_matrix, hidden_dim=hidden_dim, num_layers=2, dropout=dropout_rate).to(device)
            # print('RNN model initialized.')

        elif lm_type == 'l':
            model = LSTMModel(embedding_matrix, hidden_dim=hidden_dim, num_layers=2, dropout=dropout_rate).to(device)
            # print('LSTM model initialized.')
        
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Choose optimizer
        if optimizer_choice == "adam":
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Train the model
        train_seq(model, optimizer, loss_fn, len(vocab), train_loader, val_loader, num_epochs=num_epochs, early_stop_patience=3)

        # Load best model and evaluate on validation set
        model.load_state_dict(torch.load('/kaggle/working/best_model2.pth'))
        val_perplexity = calculate_perplexity_seq(model, val_loader, vocab)

        return val_perplexity  # Optuna minimizes perplexity

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    # Return best hyperparameters
    return study.best_params


def predict_top_k_words_seq(model, input_sentence, vocab, index_to_word, k=5):
    """
    Predict the top-k next words using an RNN-based language model.

    Args:
        model (nn.Module): The trained RNN language model.
        input_sentence (str): The input sentence.
        vocab (dict): Vocabulary mapping words to indices.
        index_to_word (dict): Index-to-word mapping.
        k (int): Number of top predictions to fetch.

    Returns:
        List[str]: Top-k predicted words.
    """
    model.eval()  # Set model to evaluation mode
    
    # Tokenize and encode the sentence
    input_sentence = input_sentence.lower()
    input_tokens = input_sentence.split()
    if not input_tokens:
        print("Input sentence cannot be empty.")
        return []

    input_tokens = ['SOS'] + input_tokens
    
    # Encode the entire sentence
    encoded_input = [vocab.get(word, vocab['<UNK>']) for word in input_tokens]
    input_tensor = torch.tensor([encoded_input], dtype=torch.long).to(device)

    # Lengths tensor (needed for pack_padded_sequence)
    lengths = torch.tensor([len(encoded_input)])

    # word_to_find = "headache"
    word_to_find = 'widow'

    # Ensure the word exists in the vocabulary, otherwise use <UNK>
    word_index = vocab.get(word_to_find, vocab["<UNK>"])
    print('Word index:', word_index)

    with torch.no_grad():
        # Get model predictions
        output, _ = model(input_tensor, lengths)

        # Extract the last time step's output
        last_timestep_output = output[:, -1, :]  # ✅ Last word's output

        # Apply softmax over vocabulary
        output_probs = torch.softmax(last_timestep_output, dim=-1).squeeze(0)  # Remove batch dim
        
        unk_index = vocab["<UNK>"]

        # Compute total probability excluding <UNK>
        valid_probs = output_probs.clone()
        valid_probs[unk_index] = 0  # Set <UNK> probability to zero
        total_valid_prob = valid_probs.sum().item()  # Compute valid probability sum

        # Get the top-k word indices
        top_k = torch.topk(output_probs, k + 1)  # Fetch extra in case <UNK> appears
        top_k_indices = top_k.indices.tolist()
        top_k_probs = top_k.values.tolist()

        # Filter out <UNK> and normalize using total valid probability
        filtered_predictions = [
            (index_to_word[idx], prob / total_valid_prob)  # Normalize with full valid sum
            for idx, prob in zip(top_k_indices, top_k_probs) if idx != unk_index
        ]

        # Probability of the specific word
        word_prob = (output_probs[word_index].item() / total_valid_prob) if word_index != unk_index else 0.0

    print(f"Proba('{word_to_find}'): {word_prob:.6f}")

    return filtered_predictions[:k]

def get_best_params_and_model(corpus_path, lm_type, N=None):
    path_dict = {
        './corpora/Pride and Prejudice - Jane Austen.txt': 1, 
        './corpora/Ulysses - James Joyce.txt': 2
    }
    model_dict = {'f': 'fnn', 'r': 'rnn', 'l': 'lstm'}
    
    model_name = f'best_model_corp{str(path_dict[corpus_path])}_{model_dict[lm_type]}'
    if lm_type == 'f':
        model_name += f'_N{N}'
    model_name += '.pth'

    params_dict = {
        (1, 'f', 3): {'batch_size': 32, 'hidden_dim': 256, 'dropout': 0.2, 'optimizer': 'sgd'}, # done 
        (1, 'f', 5): {'batch_size': 32, 'hidden_dim': 256, 'dropout': 0.5, 'optimizer': 'sgd'}, # done
        (2, 'f', 3): {'batch_size': 32, 'hidden_dim': 256, 'dropout': 0.5, 'optimizer': 'sgd'}, # done
        (2, 'f', 5): {'batch_size': 32, 'hidden_dim': 256, 'dropout': 0.5, 'optimizer': 'sgd'}, # done
        (1, 'r', None): {'batch_size': 64, 'hidden_dim': 128, 'dropout': 0.5, 'optimizer': 'adam'}, # done
        (1, 'l', None): {'batch_size': 64, 'hidden_dim': 256, 'dropout': 0.2, 'optimizer': 'adam'}, # done
        (2, 'r', None): {'batch_size': 32, 'hidden_dim': 256, 'dropout': 0.5, 'optimizer': 'sgd'}, # done
        (2, 'l', None): {'batch_size': 64, 'hidden_dim': 128, 'dropout': 0.2, 'optimizer': 'adam'}, 
    }
    return params_dict[(path_dict[corpus_path], lm_type, N)], model_name

def get_ppl_file_path(fname, model_name):
    return './' + fname + '_ppl_' + "_".join(model_name.split('.')[0].split('_')[2: ]) + '.txt'

def main(lm_type, N, corpus_path, k):
    # N = None
    # if lm_type == 'f':
    #     N = int(input('Enter the length of context to use: ')) + 1

    input_sentence = str(input('Input sentence: '))

    path_dict = {
        './corpora/Pride and Prejudice - Jane Austen.txt': 1, 
        './corpora/Ulysses - James Joyce.txt': 2
    }
    
    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    if path_dict[corpus_path] == 1:
        tokenized_corpus = clean_and_tokenize_1(corpus)
    elif path_dict[corpus_path] == 2:
        tokenized_corpus = clean_and_tokenize_2(corpus)
    
    sentences = [sent.split() for sent in tokenized_corpus.split("\n")]
    # print('number of sentences:', len(sentences))

    train_sentences, test_sentences = sentences[1000: ], sentences[: 1000]
    # train_sentences, test_sentences = train_test_split(sentences, test_size=1000, random_state=42)
    if path_dict[corpus_path] == 1:
        # train_sentences, val_sentences = train_test_split(train_sentences, test_size=800, random_state=42)
        train_sentences, val_sentences = train_sentences[800: ], train_sentences[: 800]
    elif path_dict[corpus_path] == 2:
        # train_sentences, val_sentences = train_test_split(train_sentences, test_size=2000, random_state=42)
        train_sentences, val_sentences = train_sentences[2000: ], train_sentences[: 2000]

    glove_path = './archive/glove.6B.300d.txt'
    glove_dict = load_glove(glove_path)

    if lm_type == 'f':
        vocab = build_vocab(train_sentences, glove_dict)
    
        train_inputs, train_labels = build_ngrams(train_sentences, vocab, n=N)
        test_inputs, test_labels = build_ngrams(test_sentences, vocab, n=N)
        val_inputs, val_labels = build_ngrams(val_sentences, vocab, n=N)
    
        train_inputs = torch.LongTensor(train_inputs).to(device)
        train_labels = torch.LongTensor(train_labels).to(device)
        val_inputs = torch.LongTensor(val_inputs).to(device)
        val_labels = torch.LongTensor(val_labels).to(device)
        test_inputs = torch.LongTensor(test_inputs).to(device)
        test_labels = torch.LongTensor(test_labels).to(device)
    
        train_dataset = TensorDataset(train_inputs, train_labels)
        val_dataset = TensorDataset(val_inputs, val_labels)
        test_dataset = TensorDataset(test_inputs, test_labels)

        # Optimize hyperparameters using Optuna
        # print('Hyper-param optimization started... ')
        # best_params = optimize_hyperparams(train_dataset, val_dataset, test_dataset, vocab, glove_dict, n=N, num_trials=15, num_epochs=8)

        best_params, model_name = get_best_params_and_model(corpus_path, lm_type, N)
        # print("Best Hyperparameters:", best_params)
        # print('Best model: ', model_name)
    
        # Use best hyperparameters to train the final model
        batch_size = best_params["batch_size"]
        hidden_dim = best_params["hidden_dim"]
        dropout_rate = best_params["dropout"]
        optimizer_choice = best_params["optimizer"]
        embedding_dim = 300

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        embedding_matrix = build_embedding_matrix(vocab, glove_dict, embedding_dim)
        
        model = FFNN(embedding_dim, hidden_dim, len(vocab), embedding_matrix, n=N, dropout=dropout_rate).to(device)
        print('FFNN model initialized.')
        # loss_fn = nn.CrossEntropyLoss().to(device)

        # if optimizer_choice == "adam":
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
        # print("Training model...")
        # train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=20)
        # model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
        model.load_state_dict(torch.load(f'./models/{model_name}', map_location=torch.device('cpu')))

        # Evaluate Perplexity
        # test_ppl_file_path = get_ppl_file_path('test', model_name)
        # test_perplexity = calculate_perplexities_and_save(model, test_loader, test_sentences, vocab, test_ppl_file_path)
        # print(f"Test Perplexity: {test_perplexity:.2f}")

        # val_ppl_file_path = get_ppl_file_path('val', model_name)
        # validation_perplexity = calculate_perplexities_and_save(model, val_loader, val_sentences, vocab, val_ppl_file_path) # calculate_perplexity(model, val_loader)
        # print(f"Validation Perplexity: {validation_perplexity:.2f}")
    
        # train_ppl_file_path = get_ppl_file_path('train', model_name)
        # train_perplexity = calculate_perplexities_and_save(model, train_loader, train_sentences, vocab, train_ppl_file_path)
        # print(f"Train Perplexity: {train_perplexity:.2f}")
    
        index_to_word = {index: word for word, index in vocab.items()}
        predictions = predict_top_k_words(model, input_sentence, vocab, index_to_word, k, N)

        print("\nTop-k Predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"{i}. {word} {prob:.4f}")

    else:
        vocab = build_vocab_seq(train_sentences, glove_dict)
        # embedding_matrix = [glove_dict.get(word, [0]*300) for word in vocab]
        embedding_matrix = build_embedding_matrix(vocab, glove_dict, 300)
        
        train_dataset = TextDataset(train_sentences, vocab, embedding_matrix)
        val_dataset = TextDataset(val_sentences, vocab, embedding_matrix)
        test_dataset = TextDataset(test_sentences, vocab, embedding_matrix)

        # Optimize hyperparameters using Optuna
        # print('Hyper-param optimization started... ')
        # best_params = optimize_hyperparams_seq(lm_type, train_dataset, val_dataset, test_dataset, vocab, glove_dict, num_trials=20, num_epochs=8)
        best_params, model_name = get_best_params_and_model(corpus_path, lm_type, N=None)
        # print("Best Hyperparameters:", best_params)
        # print('Best model: ', model_name)

        # Use best hyperparameters to train the final model
        batch_size = best_params["batch_size"]
        hidden_dim = best_params["hidden_dim"]
        dropout_rate = best_params["dropout"]
        optimizer_choice = best_params["optimizer"]
        
        # Use partial to pass vocab when calling DataLoader
        collate_fn_seq = get_collate_fn(vocab)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_seq)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_seq)

        if lm_type == 'r':
            model = RNNModel(embedding_matrix, hidden_dim=hidden_dim, num_layers=2, dropout=dropout_rate).to(device)
            print('RNN model initialized.')

        elif lm_type == 'l':
            model = LSTMModel(embedding_matrix, hidden_dim=hidden_dim, num_layers=2, dropout=dropout_rate).to(device)
            print('LSTM model initialized.')
        
        # loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

        # if optimizer_choice == "adam":
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
        # print('Model training started.')
        # train_seq(model, optimizer, loss_fn, len(vocab), train_loader, val_loader, num_epochs=20, early_stop_patience=3)
        # model.load_state_dict(torch.load('/kaggle/working/best_model2.pth'))
        model.load_state_dict(torch.load(f'./models/{model_name}', map_location=torch.device('cpu')))

        # Evaluate Perplexity
        # test_ppl_file_path = get_ppl_file_path('test', model_name)
        # test_perplexity = calculate_perplexities_and_save_seq(model, test_loader, test_sentences, vocab, test_ppl_file_path)
        # print(f"Test Perplexity: {test_perplexity:.2f}")

        # val_ppl_file_path = get_ppl_file_path('val', model_name)
        # validation_perplexity = calculate_perplexities_and_save_seq(model, val_loader, val_sentences, vocab, val_ppl_file_path) # calculate_perplexity(model, val_loader)
        # print(f"Validation Perplexity: {validation_perplexity:.2f}")
    
        # train_ppl_file_path = get_ppl_file_path('train', model_name)
        # train_perplexity = calculate_perplexities_and_save_seq(model, train_loader, train_sentences, vocab, train_ppl_file_path)
        # print(f"Train Perplexity: {train_perplexity:.2f}")

        index_to_word = {index: word for word, index in vocab.items()}
        top_k_predictions = predict_top_k_words_seq(model, input_sentence, vocab, index_to_word, k=k)
        
        # Print results
        print(f"\nTop-{k} Next Word Predictions:")
        for i, (word, prob) in enumerate(top_k_predictions, 1):
            print(f"{i}. {word} {prob:.4f}")
        
    # return round(train_perplexity, 2), round(validation_perplexity, 2), round(test_perplexity, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lm_type")
    parser.add_argument("N", type=int)
    parser.add_argument("corpus_path")
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    
    main(args.lm_type, args.N, args.corpus_path, args.k)

"""
### Code to create and save perplexity files and report average perplexities for all combinations ###

results = []

lm_types = ['f', 'r', 'l'] 
Ns = [3, 5, None]
corpus_paths = [
    './corpora/Pride and Prejudice - Jane Austen.txt',
    './corpora/Ulysses - James Joyce.txt'
]

# Iterate over all possible combinations of parameters
for lm_type, N, corpus_path in itertools.product(lm_types, Ns, corpus_paths):
    if (lm_type == 'f' and not N) or (lm_type != 'f' and N):  
        continue
        
    train_ppl, val_ppl, test_ppl = main(lm_type, N, corpus_path, 5)
    
    model_name = ''
    if lm_type == 'f':
        model_name += f'FFNN(n={N}) - '
    elif lm_type == 'r':
        model_name += 'RNN - '
    else: 
        model_name += 'LSTM - '
    model_name += corpus_path.split('/')[-1].split('-')[0]
    # print(model_name)
    results.append([model_name, train_ppl, val_ppl, test_ppl])

# Create a DataFrame with proper column names
df = pd.DataFrame(results, columns=["Model", "Train PPL", "Validation PPL", "Test PPL"])
print(df)

"""