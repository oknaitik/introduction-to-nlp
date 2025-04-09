import nltk
# nltk.data.path.append("C:\\Users\\NAITIK\\AppData\\Roaming\\nltk_data")

from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources
nltk.download('brown')
nltk.download('punkt_tab')

# Load and preprocess the Brown Corpus
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