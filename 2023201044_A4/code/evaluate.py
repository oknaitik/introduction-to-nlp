import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        model: Trained classifier model.
        dataloader: DataLoader for test or train set.
        device: CPU or GPU.
    
    Returns:
        None (prints evaluation metrics)
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations
        for texts, lengths, labels in dataloader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)

            # Forward pass
            outputs = model(texts, lengths)

            # Get predicted labels
            preds = torch.argmax(outputs, dim=1)  # Select the class with highest probability

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["World", "Sports", "Business", "Sci/Tech"]))

    # Compute and print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")