## 1. Model Training and Probability of Sentence:

To train an n-gram language model and compute sentence probability, use:
```
python3 language_model.py <lm_type> <N> <corpus_path>
```

Arguments:
* <lm_type>: Smoothing technique:
    * l → Laplace smoothing
    * g → Good-Turing smoothing
    * i → Linear interpolation
    
* <N>: The order of the n-gram model (e.g., 2 for bigram, 3 for trigram)
* <corpus_path>: Path to the text corpus used for training

## 2. Next Word Generation: 

To generate the top-k next-word predictions using the trained language model, run:
```
python3 generator.py <lm_type> <N> <corpus_path> <k>
```

Arguments:
* <lm_type>: Smoothing technique:
    * l → Laplace smoothing
    * g → Good-Turing smoothing
    * i → Linear interpolation
    
* <N>: The order of the n-gram model (e.g., 2 for bigram, 3 for trigram)
* <corpus_path>: Path to the text corpus used for training
* <k>: Number of top next-word predictions to generate




