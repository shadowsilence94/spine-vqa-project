from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, hamming_loss, classification_report
import numpy as np

def compute_classification_metrics(targets, predictions, classes):
    """
    Computes standard multiclass metrics (accuracy, precision, recall, F1, confusion matrix)
    for disease classification.
    """
    accuracy = accuracy_score(targets, predictions)
    
    # Calculate macro and weighted precision, recall, and F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    
    report = classification_report(
        targets, predictions, target_names=classes, labels=list(range(len(classes))), zero_division=0, output_dict=True
    )
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report
    }

def compute_localization_metrics(targets, predictions, levels):
    """
    Computes multi-label metrics (subset accuracy, hamming loss, precision, recall, F1)
    for vertebral level localization.
    """
    # Subset accuracy (exact match ratio)
    exact_match = np.all(targets == predictions, axis=1).mean()
    
    # Hamming loss (fraction of incorrect labels)
    h_loss = hamming_loss(targets, predictions)
    
    # Per-label accuracy
    num_samples = targets.shape[0]
    num_labels = targets.shape[1]
    per_label_acc = {}
    for i in range(num_labels):
        level_name = levels[i]
        label_acc = (targets[:, i] == predictions[:, i]).mean()
        per_label_acc[level_name] = label_acc
        
    # Micro and macro F1 scores
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    
    return {
        "exact_match_accuracy": exact_match,
        "hamming_loss": h_loss,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "per_level_accuracy": per_label_acc
    }
