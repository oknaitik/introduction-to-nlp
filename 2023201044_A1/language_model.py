import re
from collections import Counter, defaultdict
import random
import math
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import linregress
import sys

# Function to load corpus
def load_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

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

def generate_ngrams(tokens, N):
    """Generate N-grams from a list of tokens."""
    ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens)-N+1)]
    return ngrams

def build_ngram_model(sentences, N):
    """
    Build an N-gram model (N=1,3,5) from the corpus.
    Returns n-gram probabilities and counts.
    """
    # Tokenize corpus using previous function
    # tokenized_corpus = clean_and_tokenize(corpus)  
    # sentences = tokenized_corpus.split("\n")  # Sentence-wise processing
    all_tokens = []
    
    for sentence in sentences:
        tokens = sentence.split()
        if N-2 > 0: # add extra N-2 SOS tokens
            start_token = 'SOS'
            tokens = [start_token]*(N-2) + tokens
        # if not all_tokens: 
        #     print('Tokens: ', tokens) 
        all_tokens.extend(tokens)

    # Get N-grams
    ngrams = generate_ngrams(all_tokens, N)
    total_ngrams = len(ngrams)
    
    # Count N-gram frequencies
    ngram_counts = Counter(ngrams)
    # print('Count of ngrams with freq 1:', Counter(ngram_counts.values())[1], total_ngrams)
    
    # Count (N-1)-gram frequencies (for probability calculations)
    n_minus_1_grams = generate_ngrams(all_tokens, N-1)
    n_minus_1_counts = Counter(n_minus_1_grams)
    
    return ngram_counts, n_minus_1_counts, total_ngrams


def no_smoothing(ngram_counts, n_minus_1_counts, N):
    smoothed_probs = defaultdict(float)
    
    for ngram in ngram_counts: 
        if N == 1: 
            smoothed_probs[ngram] = ngram_counts[ngram]/ sum(ngram_counts.values())
            continue
        
        prefix = ngram[: -1]
        smoothed_probs[ngram] = ngram_counts[ngram]/ n_minus_1_counts[prefix]
    smoothed_probs['UNK'] = 1e-8 # epsilon probability assigned
    
    return smoothed_probs

def laplace_smoothing(ngram_counts, n_minus_1_counts, vocab_size, N):
    """Compute Laplace-smoothed probabilities for N-grams."""
    smoothed_counts = defaultdict(float)
    # print(N, vocab_size)
    
    for ngram in ngram_counts:
        if N == 1: 
            smoothed_counts[ngram] = ngram_counts[ngram] #/ sum(ngram_counts.values())
            continue
        prefix = ngram[:-1]
        smoothed_counts[ngram] = (ngram_counts[ngram] + 1) / (n_minus_1_counts[prefix] + vocab_size) * n_minus_1_counts[prefix]
    smoothed_counts['UNK'] = 1 # / vocab_size * smoothed_counts[ngram] # if ngram is absent, the counts equal zero.
    
    return smoothed_counts


def get_r_star_T(freq_of_freqs): 
    r_star_T, var_r_star_T = {}, {}
    
    for r, Nr in freq_of_freqs.items(): 
        Nr_plus1 = freq_of_freqs.get(r+1, 0)
        r_star_T[r] = Nr
        var_r_star_T[r] = ((r + 1)**2) * (Nr_plus1/ (Nr**2)) * (1 + Nr_plus1/ Nr)
    return r_star_T, var_r_star_T

def get_r_star_LGT(freq_of_freqs):
    # Sort dictionary by r
    sorted_freqs = sorted(freq_of_freqs.items())  
    r_values, N_values = zip(*sorted_freqs)  

    # Convert to log scale (ignore zero values to avoid log(0) errors)
    r_values = np.array(r_values)

    Z_values = np.array(N_values, dtype=np.float64)
    for i in range(len(Z_values)):
        if i == len(Z_values)-1: 
            Z_values[i] = Z_values[i]/ (r_values[i] - r_values[i-1])
        else:
            Z_values[i] = 2 * Z_values[i]/ ((r_values[i+1]) - (r_values[i-1] if i > 0 else 0))
    log_Z = np.log(Z_values)
    log_r = np.log(r_values)

    # Fit log(Z) = a + b * log(r) using linear regression
    slope, intercept, _, _, _ = linregress(log_r, log_Z)

    r_star_LGT = {r: np.exp(intercept + slope * np.log(r)) for r in range(1, max(freq_of_freqs)+1)}
    return r_star_LGT

def best_estimate(freq_of_freqs):
    total_ngrams = sum([r * freq_of_freqs[r] for r in freq_of_freqs])
    N1 = freq_of_freqs[1]
    
    r_star_T, var_r_star_T = get_r_star_T(freq_of_freqs)
    r_star_LGT = get_r_star_LGT(freq_of_freqs)

    r_star_best = {}
    flag = False
    for r in sorted(freq_of_freqs):
        if flag or abs(r_star_T[r] - r_star_LGT[r]) <= 1.65 * np.sqrt(var_r_star_T[r]):
            r_star_best[r] = r_star_LGT[r]
            flag = True
            # print(r)
        else: 
            r_star_best[r] = r_star_T[r]

    p_unnorm = {}
    for r, r_star in r_star_best.items(): 
        r_next = r + 1 
        if r_next in r_star_best: 
            p_unnorm[r] = (r_next * r_star_best[r_next])/ (total_ngrams * r_star) if r_star != 0 else 0 
        else: 
            p_unnorm[r] = 1e-6

    sum_p_unnorm = sum(p_unnorm.values())
    probs = {}
    probs[0] = N1/ total_ngrams
    for r in p_unnorm:
        probs[r] = (1 - probs[0]) * p_unnorm[r]/ sum_p_unnorm 
    
    return probs

def good_turing_smoothing(ngram_counts):
    """Apply Good-Turing Smoothing to N-grams."""
    N = sum(ngram_counts.values())
    freq_of_freqs = Counter(ngram_counts.values())  # Frequency of frequencies
    probs = best_estimate(freq_of_freqs)

    # print(probs[0])
    adjusted_counts = {'UNK': probs[0] }
    
    for ngram, count in ngram_counts.items():
        adjusted_counts[ngram] = probs[count]
    return adjusted_counts

def compute_lambdas(held_out_sentences, N):
    """
    Compute lambda values for linear interpolation using a held-out corpus.
    
    Args:
        held_out_corpus (list of str): List of sentences (already tokenized and containing SOS/EOS markers).
        N (int): Maximum N-gram order.
    
    Returns:
        list: Normalized lambda values [L1, L2, ..., LN].
    """
    
    counts_dict = {} # stores ngram counts for all possible n
    for n in range(1, N+1):
        ngram_counts, _, _ = build_ngram_model(held_out_sentences, n)
        # add ngram_counts to counts_dict
        counts_dict[n] = ngram_counts
    
    # Initialize lambda values
    lambdas = [0] * N

    # Compute lambda assignments
    for ngram in counts_dict[N]:  # Iterate over full N-grams
        max_prob = (counts_dict[1][ngram[-1]] - 1)/ (sum(counts_dict[1].values()) - 1)
        best_n = 1

        for n in range(2, N+1):
            current = ngram[-n: ]
            current_count = counts_dict[n][current] - 1
            prev = current[: -1]
            prev_count = counts_dict[n-1][prev] - 1
            if prev_count > 0:
                prob = current_count/ prev_count
                if prob > max_prob:
                    max_prob = prob
                    best_n = n

        # Assign to the best lambda (1-based index)
        lambdas[best_n-1] += counts_dict[N][ngram]  # Weight by N-gram count

    # Normalize lambda values
    total_lambda = sum(lambdas)
    if total_lambda > 0: 
        lambdas = [l/ total_lambda for l in lambdas]
    else: 
        lambdas = [1/N] * N  # Prevent division by zero
    
    return lambdas

def linear_interpolation(held_out_sentences, N):
    """Apply Linear Interpolation smoothing to an N-gram model."""
    smoothed_probs = defaultdict(float)
    lambdas = compute_lambdas(held_out_sentences, N)
    # print(lambdas)
    
    return lambdas


def train_language_model(train_sentences, lm_type, N):
    """Train N-gram models for N=1,3,5 with different smoothing techniques."""
    models = {}

    # Split the sentences into train and dev sets
    num_dev_sentences = 200
    held_out_sentences = train_sentences[: num_dev_sentences]
    train_sentences = train_sentences[num_dev_sentences: ]
    
    ngram_counts, n_minus_1_counts, total_ngrams = build_ngram_model(train_sentences, N=1)
    vocab_size = len(ngram_counts)

    """ 
    if type = 'linear', then store the counts of ngrams for all n 
    if type = 'laplace', then store just the counts of n grams, n-1 grams 
    """
    
    counts_dict = {}
    if lm_type == 'na':
        for n in range(1, N+1):
            ngram_counts, n_minus_1_counts, total_ngrams = build_ngram_model(train_sentences, N=n)
            # ngram_counts['UNK'] = 1
            counts_dict[n] = ngram_counts
        return counts_dict
    
    elif lm_type == 'l':
        for n in range(1, N+1):
            ngram_counts, n_minus_1_counts, total_ngrams = build_ngram_model(train_sentences, N=n)
            counts_dict[n] = laplace_smoothing(ngram_counts, n_minus_1_counts, vocab_size, n)
        return counts_dict
    
    elif lm_type == 'g':
        ngram_counts, n_minus_1_counts, total_ngrams = build_ngram_model(train_sentences, N)
        counts_dict[N] = good_turing_smoothing(ngram_counts) # actually probability dictionary
        return counts_dict
    
    elif lm_type == 'i':
        lambdas = linear_interpolation(held_out_sentences, N)
        for n in range(1, N+1):
            ngram_counts, n_minus_1_counts, total_ngrams = build_ngram_model(train_sentences, N=n)
            counts_dict[n] = ngram_counts
        # print(lambdas)
        return lambdas, counts_dict

def sentence_perplexity(sentence, counts_dict, N, smoothing, lambdas=None, generate=False): 
    tokens = re.findall(r"\w+|[.,!?\"']", sentence)
    # print(tokens)
    
    start_token, end_token = 'SOS', 'EOS'
    if tokens[0] != start_token:
        tokens = [start_token]*(N-1) + tokens  
    else: # add N-2 times
        tokens = [start_token]*max(N-2, 0) + tokens

    if not generate: 
        if tokens[-1] != end_token: # add EOS once
            tokens = tokens + [end_token] 

    # m = len(tokens) - (N-1) # exclude all SOS tokens (equivalent to number of ngrams)
    ngrams = generate_ngrams(tokens, N)
    # print('Ngram:', ngrams)
    
    if len(ngrams) == 0:  
        return 0
        
    # log_proba_sum = 0
    # for ngram in ngrams: 
    #     # print(ngram, ngram in model_probs)
    #     print('Ngram:', ngram) 
        
    #     if ngram in model_probs:
    #         print(ngram, model_probs[ngram])
    #         log_proba_sum += math.log2(model_probs[ngram])
    #     else:
    #         print('Unk:', model_probs['UNK'])
    #         log_proba_sum += math.log2(model_probs['UNK'])
    
    log_proba_sum = 0
    
    eps = 1e-5
    if smoothing == 'no_smoothing':
        for ngram in ngrams:
            if ngram not in counts_dict[N]: 
                log_proba_sum += math.log2(eps)  
                continue
            
            prob = eps
            if N == 1:
                prob = max(prob, counts_dict[N][ngram]/ sum(counts_dict[1].values()))
            else: 
                prefix = ngram[:-1]
                if prefix in counts_dict[N-1]:
                    prob = max(prob, counts_dict[N][ngram]/ counts_dict[N-1][prefix])
            
            log_proba_sum += math.log2(prob) 


    elif smoothing == 'laplace': 
        for ngram in ngrams:
            if ngram not in counts_dict[N]: 
                # print(f'{ngram} not in Counts matrix')
                log_proba_sum += math.log2(1/ len(counts_dict[1]))  # 1/|V|
                continue

            prob = eps
            if N == 1:
                prob = max(prob, counts_dict[N][ngram]/ sum(counts_dict[1].values()))
            else: 
                prefix = ngram[:-1]
                if prefix in counts_dict[N-1]:
                    prob = max(prob, counts_dict[N][ngram]/ counts_dict[N-1][prefix])

            # print(f'Prob of {ngram} is {prob}')
            log_proba_sum += math.log2(prob)   
            
    elif smoothing == 'good_turing':
        for ngram in ngrams: 
            if ngram in counts_dict[N]:
                # print(f'Prob of {ngram} is {counts_dict[N][ngram]}')
                log_proba_sum += math.log2(counts_dict[N][ngram])
            else: 
                # print(f'{ngram} not in Counts matrix')
                log_proba_sum += math.log2(counts_dict[N]['UNK'])

    elif smoothing == 'interpolation':   
        total_unigrams = sum(counts_dict[1].values())

        for ngram in ngrams: 
            prob = 0  # Initialize probability for the n-gram
    
            # Compute probability using linear interpolation
            for i in range(N):
                prev_ngram = ngram[-(i+1):]  # Get last (i+1)-gram
                prefix = prev_ngram[:-1]
    
                numerator = counts_dict[len(prev_ngram)].get(prev_ngram, 0)
                if len(prefix) == 0:  # Unigram case
                    denominator = total_unigrams  # Sum of all unigrams
                else:  # Higher order n-grams
                    denominator = counts_dict[len(prefix)].get(prefix, 0)
    
                # Add smoothed probability
                if denominator > 0:
                    prob += lambdas[i] * (numerator / denominator)
            
            if prob == 0:  # If all probabilities are zero, fall back to UNK
                prob = lambdas[0] * (1 / total_unigrams)
    
            log_proba_sum += math.log2(prob)     
    
    # print('Log sum:', log_proba_sum)
    # print(len(ngrams), math.pow(2, -1/ len(ngrams) * log_proba_sum))
    return len(ngrams), math.pow(2, -1/ len(ngrams) * log_proba_sum)

def sentence_proba(sentence, counts_dict, N, smoothing, lambdas=None, generate=False):
    m, ppl = sentence_perplexity(sentence, counts_dict, N, smoothing, lambdas, generate)
    # print("Length, PPL for the sentence:", m, ppl)
    return math.pow(ppl, -m)

def perplexity(sentences, trained_params, lm_type, N, output_filename):
    """Compute Perplexity of a given model on test sentences."""
    perplexities = []
    results = []

    for sentence in sentences: 
        if lm_type == 'l':
            _, perplexity = sentence_perplexity(sentence, trained_params, N, smoothing='laplace')
        elif lm_type == 'g':
            _, perplexity = sentence_perplexity(sentence, trained_params, N, smoothing='good_turing')
        elif lm_type == 'i':
            _, perplexity = sentence_perplexity(sentence, trained_params[1], N, smoothing='interpolation', lambdas=trained_params[0])
        elif lm_type == 'na':
            _, perplexity = sentence_perplexity(sentence, trained_params, N, smoothing='no_smoothing')
        
        perplexities.append(perplexity)
        results.append(f"{sentence}\t{perplexity:.6f}")
        
    avg_perplexity = sum(perplexities) / len(perplexities)
    
    lm_map = {'l': 0, 'g': 1, 'i': 2}
    x = int(output_filename.split('_')[1][-1]) + lm_map[lm_type]
    output_filename = '_'.join([output_filename.split('_')[0]] + ['LM' + str(x)] + output_filename.split('_')[2: ])
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"{avg_perplexity:.6f}\n")  # Write average perplexity in the first line
        f.write("\n".join(results))  # Write each sentence with its perplexity

    return avg_perplexity


def create_train_test_sets(sentences, num_test_sentences=1000):
    """Create mutually exclusive train and test sets."""
    # Shuffle the sentences to randomize
    # random.shuffle(sentences)
    
    # Split the sentences into train and test sets
    test_sentences = sentences[:num_test_sentences]
    train_sentences = sentences[num_test_sentences:]
    
    return train_sentences, test_sentences


# Main function to run the model based on command-line args
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 language_model.py <lm_type> <N> <corpus_path>")
        sys.exit(1)

    # Get command line arguments
    lm_type = sys.argv[1]  # 'l' for Laplace, 'g' for Good-Turing, 'i' for Interpolation
    N = int(sys.argv[2])
    corpus_path = sys.argv[3]

    filename = '2023201044_LM'
    # Train appropriate model
    if corpus_path == 'Pride and Prejudice - Jane Austen.txt':
        # Load the corpus
        corpus = load_corpus(corpus_path)        
        # Tokenize the corpus
        tokenized_corpus = clean_and_tokenize_1(corpus)
        filename += '1_'

    elif corpus_path == 'Ulysses - James Joyce.txt':
        corpus = load_corpus(corpus_path)

        # Tokenize the corpus
        tokenized_corpus = clean_and_tokenize_2(corpus)
        filename += '4_'
        
    else: 
        print('Inappropriate corpus path.')
        sys.exit(1)

    filename += f'{N}_'
    sentences = tokenized_corpus.split("\n")
    print('Total sentences:', len(sentences))
    
    train_sentences, test_sentences = create_train_test_sets(sentences, num_test_sentences=1000)
    trained_params = train_language_model(train_sentences, lm_type, N)
    # print(test_sentences[0])

    # <roll number>_LM1_N_train-perplexity.txt
    print('Average PPL on Train:', perplexity(train_sentences, trained_params, lm_type, N, filename + 'train-perplexity.txt'))
    print('Average PPL on Test:', perplexity(test_sentences, trained_params, lm_type, N, filename + 'test-perplexity.txt'))

    # Interactive prompt for user input sentence
    sentence = input("Enter a sentence: ")  
    
    # # # Compute sentence probability
    sentence_probability = 1.0 
    sentence = clean_and_tokenize_1(sentence)
    
    if lm_type == 'l':
        sentence_probability = sentence_proba(sentence, trained_params, N, smoothing='laplace')
    elif lm_type == 'g':
        sentence_probability = sentence_proba(sentence, trained_params, N, smoothing='good_turing')
    elif lm_type == 'i':
        sentence_probability = sentence_proba(sentence, trained_params[1], N, smoothing='interpolation', lambdas=trained_params[0])
    elif lm_type == 'na':
        sentence_probability = sentence_proba(sentence, trained_params, N, smoothing='no_smoothing')
    else:
        print('Inappropriate Smoothing method.')
        sys.exit(1)

    print(f"Probability of the sentence: {sentence_probability}")


# Entry point for running the script
if __name__ == "__main__":
    main()
