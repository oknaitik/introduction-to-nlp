import re
from html import unescape

def clean_text(text):
    """
    Cleans text by:
    - Converting HTML entities
    - Removing URLs and email addresses
    - Removing backslashes, digits, and punctuation
    - Normalizing whitespace
    - Converting to lowercase
    """
    # Convert HTML entities (e.g., &lt; becomes <)
    text = unescape(text)
    
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r"\\+", " ", text)  # Remove unnecessary backslashes

    # Check if there is a hyphen in the text; if yes, see if the prefix has at most 5 words.
    if '-' in text:
        parts = text.split('-', 1)  # Split on the first hyphen.
        prefix = parts[0].strip()
        if len(prefix.split()) <= 5:
            text = parts[1].strip()
    
    text = re.sub(r"\d+", "", text)  # Remove numeric characters
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    text = text.lower()  # Convert to lowercase
    
    return text
