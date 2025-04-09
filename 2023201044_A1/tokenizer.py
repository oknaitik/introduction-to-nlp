import re

def tokenize(text):
    # This regular expression matches words and punctuation separately
    # The regex matches a sequence of word characters or individual punctuation characters.
    tokens = re.findall(r"\w+|[^\w\s]", text)
    
    # Now we will split the tokens into sentences (separated by periods, question marks, etc.)
    sentences = []
    current_sentence = []
    
    for token in tokens:
        if token in ['.', '!', '?']:  # We treat punctuation as sentence delimiters.
            current_sentence.append(token)
            sentences.append(current_sentence)
            current_sentence = []  # Start a new sentence
        else:
            current_sentence.append(token)
    
    if current_sentence:  # Add any remaining tokens that didn't end with punctuation.
        sentences.append(current_sentence)
    
    return sentences

# Example usage
if __name__ == "__main__":
    text = input("Enter your text: ")
    tokenized_text = tokenize(text)
    print("Tokenized text:")
    print(tokenized_text)

