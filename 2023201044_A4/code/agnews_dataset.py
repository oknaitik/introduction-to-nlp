import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import json
import re

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from text_utils import clean_text

import warnings
warnings.filterwarnings("ignore")


class AGNewsDataset(Dataset):
    def __init__(self, df, word2idx, max_len=100):
        # df = pd.read_csv(csv_path)  # Read CSV file

        self.data = list(zip(df["Class Index"], df["Description"]))  # Extract Class & Description
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        label -= 1 # 0-index classes
        text = clean_text(text)  # Clean the text

        tokens = word_tokenize(text)
        indices = [self.word2idx.get(token, len(self.word2idx)-2) for token in tokens]  # Convert words to indices

        # Truncate/Padding
        indices = indices[:self.max_len] + [len(self.word2idx)-1] * max(0, self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn_agnews(batch):
    texts, labels = zip(*batch)
    # Compute actual lengths before padding
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    # Pad sequences
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return texts, lengths, labels


def prepare_agnews_dataloaders(train_path, test_path, word2idx, BATCH_SIZE=64):   
    # Load the dataset
    df = pd.read_csv(train_path)
    X = df["Description"].values
    y = df["Class Index"].values
    
    # Stratified split (5000 validation samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    # Create DataFrames for train and validation
    train_df = pd.DataFrame({"Description": X_train, "Class Index": y_train})
    val_df = pd.DataFrame({"Description": X_val, "Class Index": y_val})
    
    # Load test set
    test_df = pd.read_csv("../dataset/test.csv")
    

    # Create dataset instances
    ag_train_dataset = AGNewsDataset(train_df, word2idx=word2idx, max_len=100)
    ag_val_dataset = AGNewsDataset(val_df, word2idx=word2idx, max_len=100)
    ag_test_dataset = AGNewsDataset(test_df, word2idx=word2idx, max_len=100)
    
    # Create DataLoaders
    # BATCH_SIZE = 64
    ag_train_loader = DataLoader(ag_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_agnews)
    ag_val_loader = DataLoader(ag_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_agnews)
    ag_test_loader = DataLoader(ag_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_agnews)
    
    # print(len(ag_train_loader)*BATCH_SIZE, len(ag_val_loader)*BATCH_SIZE, len(ag_test_loader)*BATCH_SIZE)
    return ag_train_loader, ag_val_loader, ag_test_loader