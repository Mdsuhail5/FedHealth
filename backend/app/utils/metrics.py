import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate detailed model metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    metrics = {
        'accuracy': (y_pred == y_true).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    return metrics 