import torch

from elmo_class import ELMo
from agnews_classifier import AGNewsClassifier
from embedding_extractor import EmbeddingExtractor

import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classifier(embedding_type, lambda_mode, embed_path, classifier_path): 
    if embedding_type == 'elmo':
        embedding_matrix = torch.load(embed_path, map_location=torch.device(device))
    
        # Instantiate the model architecture first
        if lambda_mode == "trainable":
            # print(f'I am at trainable')
            elmo_model = ELMo(embedding_matrix, lambda_mode="trainable").to(device)
            embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
            classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=True).to(device)
            
        elif lambda_mode == "frozen":
            # print(f'I am at frozen')
            elmo_model = ELMo(embedding_matrix, lambda_mode="frozen").to(device)
            embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
            classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=False).to(device)
            
        elif lambda_mode == "function":
            # print(f'I am at function')
            elmo_model = ELMo(embedding_matrix, lambda_mode="function").to(device)
            embedding_extractor = EmbeddingExtractor(embedding_type="elmo", elmo_model=elmo_model)
            classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4, fine_tune_elmo=True).to(device)
            
        else:
            raise ValueError("Invalid lambda_mode.")
    
        # Now load the state dict into the classifier
        state_dict = torch.load(classifier_path, map_location=device)

        if lambda_mode == "function":
            # print(state_dict.keys())

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("gamma_mlp."):
                    new_key = "embedding_extractor.elmo_model." + key
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            
            # print("Remapped keys in state_dict:", state_dict.keys())
            
        classifier.load_state_dict(state_dict)
        classifier.eval()
        
    else: 
        embedding_matrix = torch.load(embed_path, map_location=torch.device("cpu"))
        extended_embedding_matrix = extend_embeddings(embedding_matrix).to(device)
        embedding_extractor = EmbeddingExtractor(embedding_type=embedding_type, embedding_matrix=extended_embedding_matrix)
        classifier = AGNewsClassifier(embedding_extractor, rnn_hidden_size=256, num_classes=4).to(device)

        state_dict = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()        
    return classifier


def extend_embeddings(embedding_matrix):
    embedding_dim = embedding_matrix.shape[1]
    unk_embedding = torch.randn(1, embedding_dim, device=embedding_matrix.device)  # You can also use zeros if preferred.
    pad_embedding = torch.zeros(1, embedding_dim, device=embedding_matrix.device)
    
    extended_embedding_matrix = torch.cat([embedding_matrix, unk_embedding, pad_embedding], dim=0)
    # print("Extended SVD embedding matrix shape:", extended_embedding_matrix.shape)  # Expected: [40656, 300]

    return extended_embedding_matrix