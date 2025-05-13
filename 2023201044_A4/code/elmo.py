import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Word2Vec
from datasets import load_dataset

from elmo_class import ELMo

import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_brown():
    corpus_text = " ".join([" ".join(sent) for sent in brown.sents()])  # Merge all sentences
    sentences = sent_tokenize(corpus_text)  # Sentence tokenization
    processed_corpus = []

    for sentence in sentences:
        words = word_tokenize(sentence)  # Word tokenization
        words = [word.lower() for word in words if word.isalpha()]  # Lowercase & remove punctuation
        if words:
            processed_corpus.append(words)

    return processed_corpus
    
# Train Word2Vec for embeddings
# sentences = get_brown_sentences()
sentences = preprocess_brown()
print(len(sentences))


def build_vocab(corpus):
    # # Build vocabulary
    # word_counts = Counter(word for sentence in brown_corpus for word in sentence)
    # # Sort word_counts by frequency (descending) to keep order consistent
    # sorted_vocab = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Assign indices, reserving 0 for padding and 1 for unknown words (OOV handling)
    word2idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    dict_size = len(word2idx)
    word2idx['UNK'], idx2word[dict_size] = dict_size, 'UNK'
    dict_size = len(word2idx)
    word2idx['PAD'], idx2word[dict_size] = dict_size, 'PAD'
    
    return word2idx, idx2word

with open("../archive/sorted_vocab.json", "r") as f:
    sorted_vocab = json.load(f)

# print(sorted_vocab)
word2idx, idx2word = build_vocab(sorted_vocab)

# save word-to-index dictionary
with open("../archive/word2idx.json", "w") as f:
    json.dump(word2idx, f)


def build_embedding_matrix(embedding_dim=100, window=2, min_count=1):
    
    # Train Word2Vec on your corpus with min_count=1 and window=2.
    w2v_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, workers=4)
    
    # Build an embedding matrix in the order of word2idx.
    vocab_size = len(word2idx)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    
    for word, idx in word2idx.items():
        # If the word is not in the model's vocabulary (rare if min_count=0), you can choose to leave it as zeros or initialize it randomly.
        if word in w2v_model.wv:
            embedding_matrix[idx] = torch.tensor(w2v_model.wv[word])
        else:
            # Optionally, handle missing words here.
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix

# build using context window size of 2 and closed vocab
embedding_matrix = build_embedding_matrix(embedding_dim=100, window=2, min_count=1)
print(embedding_matrix.shape)
# save matrix
torch.save(embedding_matrix, "../archive/elmo_embedding_matrix.pt")


# Dataset Preparation
class BrownDataset(Dataset):
    def __init__(self, sentences, word2idx, seq_len=20):
        self.sentences = sentences
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.data = self.process_data()

    def process_data(self):
        data = []
        for sent in self.sentences:
            idxs = [self.word2idx[word] for word in sent if word in self.word2idx]
            for i in range(1, len(idxs)):
                context = idxs[max(0, i - self.seq_len):i]
                target = idxs[i]
                data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context = torch.tensor(context, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return context, target

def collate_fn(batch):
    contexts, targets = zip(*batch)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=0)  # Padding sequences
    targets = torch.tensor(targets, dtype=torch.long)  # Use -100 for ignored padding in loss

    return contexts, targets

# Training the ELMo Model
def train_model(model, dataloader, epochs=2, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Prepare dataset and train
dataset = BrownDataset(sentences, word2idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Initialize and Train for 15 Epochs
model = ELMo(embedding_matrix)
epochs = 5
train_model(model, dataloader, epochs=epochs)

# Save the trained model
torch.save(model.state_dict(), f"../models/elmo_model_{epochs}.pth")
print("Model saved after 5 epochs!")
