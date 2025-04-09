import sys
import re
import heapq
from collections import Counter
import math

from language_model import train_language_model, sentence_proba, load_corpus, clean_and_tokenize_1, clean_and_tokenize_2

def get_top_k_predictions(sentence, counts_dict, vocab, N, k, smoothing, lambdas=None):
    """Predicts the top-k most probable next words for a given sentence."""
    words = sentence.split()
    candidate_probs = []

    vocab = sorted(vocab)
    unique_probs = set()
    
    for word in vocab:
        if word in ['SOS', 'EOS']:
            continue
            
        new_sentence = " ".join(words + [word])
        prob = sentence_proba(new_sentence, counts_dict, N, smoothing, lambdas, generate=True)
        if prob > 0:  # Ignore zero-probability words
            candidate_probs.append((prob, word))
            unique_probs.add(prob)
        
    # print('Unique Probs:', len(set([prob for prob, _ in candidate_probs])))
    # if len(unique_probs) <= 2:
    #     return []
        
    # Calculate total probability mass to normalize
    total_prob = sum(prob for prob, _ in candidate_probs)
    normalized_probs = [(prob / total_prob, word) for prob, word in candidate_probs]

    # Get top-k words based on normalized probability
    top_k = heapq.nlargest(k, normalized_probs, key=lambda x: x[0])
    
    return top_k


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 generator.py <lm_type> <N> <corpus_path> <k>")
        sys.exit(1)

    # Parse command-line arguments
    lm_type = sys.argv[1]  # 'l' for Laplace, 'g' for Good-Turing, 'i' for Interpolation
    N = int(sys.argv[2])
    corpus_path = sys.argv[3]
    k = int(sys.argv[4])  # Number of next-word candidates

    # Train appropriate model
    if corpus_path == 'Pride and Prejudice - Jane Austen.txt':
        # Load the corpus
        corpus = load_corpus(corpus_path)        
        tokenized_corpus = clean_and_tokenize_1(corpus)

    elif corpus_path == 'Ulysses - James Joyce.txt':
        corpus = load_corpus(corpus_path)
        print('Corpus loaded.')
        tokenized_corpus = clean_and_tokenize_2(corpus)
    else: 
        print('Inappropriate corpus path.')
        sys.exit(1)
        
    sentences = tokenized_corpus.split("\n")

    # train_sentences, test_sentences = create_train_test_sets(sentences, num_test_sentences=500)
    trained_params = train_language_model(sentences, lm_type, N)
    # print(test_sentences[0])

    vocab = set()
    for sentence in sentences:
        tokens = sentence.split()
        for token in tokens:
            vocab.add(token)
    # print('Size of Vocab:', len(vocab))

    sentence = input("Input sentence: ")

    # Preprocess input sentence
    sentence = clean_and_tokenize_1(sentence)
    sentence = " ".join(sentence.split()[: -1]) # remove the EOS token 

    # Predict top-k next words
    if lm_type == 'l':
        top_k_predictions = get_top_k_predictions(sentence, trained_params, vocab, N=N, k=k, smoothing='laplace')
    elif lm_type == 'g':
        top_k_predictions = get_top_k_predictions(sentence, trained_params, vocab, N=N, k=k, smoothing='good_turing')
    elif lm_type == 'i':
        top_k_predictions = get_top_k_predictions(sentence, trained_params[1], vocab, N=N, k=k, smoothing='interpolation', lambdas=trained_params[0])
    elif lm_type == 'na':
        top_k_predictions = get_top_k_predictions(sentence, trained_params, vocab, N=N, k=k, smoothing='no_smoothing')
    else:
        print('Inappropriate Smoothing method.')
        sys.exit(1)

    print('hey')
    # if len(top_k_predictions) == 0: 
    #     print('The model predictions are weak and non-distinctive. Try some other model or smoothing technique.')
    #     return
        
    print(f"\nTop-{k} next word predictions:")
    for rank, (prob, word) in enumerate(top_k_predictions, start=1):
        print(f"{rank}. {word} (Probability: {prob})")

if __name__ == "__main__":
    main()
