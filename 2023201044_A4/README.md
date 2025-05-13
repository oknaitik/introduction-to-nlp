# ELMo Assignment

This repository contains code for training and evaluating models using various embedding techniques, including ELMo, SVD, CBOW, and Skip-gram (sgns). The assignment includes three main tasks:

1. **Pre-training the ELMo Model:** Run the ELMo model with frozen λ (lambda) parameters.
2. **Training the Classifier:** Train a classifier using the chosen embedding type.
3. **Inference:** Perform inference on a news description using a saved classifier model.

---

## Requirements

- Python 3.7 or higher
- PyTorch
- Pandas, NumPy
- NLTK (and downloaded datasets, e.g., `punkt`)
- scikit-learn
- gensim

Install the required dependencies (if not already installed):

```bash
pip install torch pandas numpy nltk scikit-learn gensim
```

*Note:* Make sure to download the NLTK `punkt` package if not already done:

```python
import nltk
nltk.download('punkt')
```

---

## Directory Structure

```
├── code/
│   ├── elmo.py            # Pre-training script for the ELMo model (frozen λ)
│   ├── classifier.py      # Script to train and save the classifier model
│   ├── inference.py       # Script to perform inference using a saved classifier model
│   └── ...                # Additional code and utility files
├── classifier/                # Folder to save trained models
└── README.md              # This file
```

---

## How to Run

### 1. Train and Save the ELMo Model

To pre-train the ELMo model (with frozen lambdas) and save it, navigate to the `code` folder and run:

```bash
python elmo.py
```

This script trains the ELMo model and saves the model parameters (in the `embeddings/` directory).

### 2. Train and Save the Classifier

To train the classifier using one of the embedding methods (ELMo, SVD, CBOW, or Skip-gram), run:

```bash
python classifier.py <embedding_type>
```

Replace `<embedding_type>` with one of the following options:
- `elmo`
- `svd`
- `cbow`
- `sgns`

For example, to train using ELMo embeddings:

```bash
python classifier.py elmo
```

The trained classifier model will be saved (typically in the `classifier/` directory).

### 3. Run Inference

To perform inference using a saved classifier model, run:

```bash
python inference.py <saved_model_path> "<description>"
```

- `<saved_model_path>`: Path to the saved classifier model.
- `<description>`: A news description (in quotes) whose class you want to predict.

For example:

```bash
python inference.py ../models/classifier_elmo_frozen.pth "The government announced new policies to boost economic growth."
```

This will output the predicted probabilities for each class.

---

## Pretrained Models
You can download pretrained models for ELMo and AGNewsClassifier from Kaggle:

[Pretrained Models on Kaggle](https://www.kaggle.com/models/naitikkariwal/inlp-a4-models/)

## Additional Information

- **ELMo Model (Frozen):**  
  The ELMo model is pre-trained with frozen λ parameters to preserve robust pre-trained representations during downstream tasks.
  
- **Classifier:**  
  The classifier leverages embeddings from different sources (ELMo, SVD, CBOW, sgns) and is built with a GRU layer followed by a fully connected layer for classification.
  
- **Inference:**  
  The inference script loads the saved classifier model and predicts the class for a given news description.

For further details on the implementation, please refer to the source code files in the `code` directory.
