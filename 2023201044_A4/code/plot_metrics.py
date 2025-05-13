import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json

from load_utils import load_classifier, extend_embeddings  # Adjust this import as needed
from agnews_dataset import prepare_agnews_dataloaders

def evaluate_model_return(model, dataloader, device):
    """
    Evaluate the model and return metrics in a dictionary.
    
    Returns:
        A dict with keys: 'accuracy' and 'f1'
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, lengths, labels in dataloader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
            outputs = model(texts, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # (Optional: print detailed report and confusion matrix)
    # print(classification_report(all_labels, all_preds))
    # print(confusion_matrix(all_labels, all_preds))
    
    return {"accuracy": accuracy, "f1": f1}

def plot_metrics(metrics_dict, title, filename):
    """
    Plots a grouped bar chart.
    
    Args:
        metrics_dict (dict): Keys are configuration names; values are dicts with 'accuracy' and 'f1'.
        title (str): Title for the plot.
        filename (str): File name to save the plot.
    """
    configs = list(metrics_dict.keys())
    accs = [metrics_dict[c]['accuracy'] for c in configs]
    f1s = [metrics_dict[c]['f1'] for c in configs]

    x = np.arange(len(configs))  # positions for each configuration
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-score')

    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()

    # Annotate bars with their values.
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

# Example usage for ELMo models:
def evaluate_elmo_models(elmo_modes, train_loader, test_loader, embed_path, classifier_path_template, device):
    """
    Evaluates the ELMo classifier for each lambda mode and returns dictionaries of metrics.
    
    Args:
        elmo_modes (list): List of lambda modes (e.g., ['trainable', 'frozen', 'function']).
        classifier_path_template (str): A format string where the lambda mode is substituted.
    Returns:
        Two dictionaries: one for train metrics, one for test metrics.
    """
    train_metrics = {}
    test_metrics = {}
    
    for mode in elmo_modes:
        classifier_path = classifier_path_template.format(mode)
        # Load the classifier using your existing function.
        classifier = load_classifier(
            embedding_type='elmo', 
            lambda_mode=mode, 
            embed_path=embed_path, 
            classifier_path=classifier_path
        )
        classifier.eval()
        print(f"Evaluating ELMo classifier ({mode}) ...")
        train_results = evaluate_model_return(classifier, train_loader, device)
        test_results = evaluate_model_return(classifier, test_loader, device)
        train_metrics[mode] = train_results
        test_metrics[mode] = test_results
    return train_metrics, test_metrics

# Similarly, for static embeddings:
def evaluate_static_models(model_types, train_loader, test_loader, embed_paths, classifier_path_template, device):
    """
    Evaluates classifiers for static embeddings (svd, cbow, sgns).
    
    Args:
        model_types (list): List of static embedding types.
        embed_paths (dict): Dictionary mapping each model type to its embedding path.
        classifier_path_template (str): Format string for classifier path.
    Returns:
        Two dictionaries: one for train metrics, one for test metrics.
    """
    # from load_classifier_module import load_classifier  # Adjust as needed
    train_metrics = {}
    test_metrics = {}
    
    for mtype in model_types:
        classifier_path = classifier_path_template.format(mtype)
        classifier = load_classifier(
            embedding_type=mtype, 
            lambda_mode=None, 
            embed_path=embed_paths[mtype], 
            classifier_path=classifier_path
        )
        classifier.eval()
        print(f"Evaluating classifier for {mtype} embeddings ...")
        train_results = evaluate_model_return(classifier, train_loader, device)
        test_results = evaluate_model_return(classifier, test_loader, device)
        train_metrics[mtype] = train_results
        test_metrics[mtype] = test_results
    return train_metrics, test_metrics

if __name__ == "__main__":
    # Example: set up your device, data loaders, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assume you have functions to prepare data loaders:
    with open("../archive/word2idx.json", "r") as f:
        word2idx = json.load(f)
    
    ag_train_loader, ag_val_loader, ag_test_loader = prepare_agnews_dataloaders(
        '../dataset/train.csv', 
        '../dataset/test.csv', 
        word2idx
    )
    
    # Define paths (adjust these as needed)
    elmo_embed_path = "../archive/elmo_embedding_matrix.pt"
    static_embed_paths = {
        "svd": "../embeddings/svd_embeddings.pt",
        "cbow": "../embeddings/cbow_embeddings.pt",
        "sgns": "../embeddings/sgns_embeddings.pt"
    }
    
    # Templates for saved classifier paths
    elmo_classifier_template = "../classifier/classifier_elmo_{}.pth"  # e.g., classifier_elmo_trainable.pth
    static_classifier_template = "../classifier/classifier_{}.pth"       # e.g., classifier_cbow.pth

    # Evaluate ELMo models
    elmo_modes = ['trainable', 'frozen', 'function']
    elmo_train_metrics, elmo_test_metrics = evaluate_elmo_models(
        elmo_modes, ag_train_loader, ag_test_loader, elmo_embed_path, elmo_classifier_template, device
    )
    
    # Create bar plots for ELMo
    plot_metrics(elmo_train_metrics, "ELMo Train Metrics", "../plots/elmo_train_metrics.png")
    plot_metrics(elmo_test_metrics, "ELMo Test Metrics", "../plots/elmo_test_metrics.png")
    
    # Evaluate static models
    static_models = ['svd', 'cbow', 'sgns']
    static_train_metrics, static_test_metrics = evaluate_static_models(
        static_models, ag_train_loader, ag_test_loader, static_embed_paths, static_classifier_template, device
    )
    # Create bar plots for static embeddings
    plot_metrics(static_train_metrics, "Static Embeddings Train Metrics", "../plots/static_train_metrics.png")
    plot_metrics(static_test_metrics, "Static Embeddings Test Metrics", "../plots/static_test_metrics.png")
