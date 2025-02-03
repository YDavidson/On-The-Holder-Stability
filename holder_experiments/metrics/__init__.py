import torch
from sklearn.metrics import average_precision_score, accuracy_score, f1_score


def setup_metric(name):
    if name == 'accuracy':
        return lambda true, pred: accuracy_score(true, pred)
    elif name == 'average_precision':
        return lambda true, pred: average_precision_score(true, pred)
    elif name == 'f1':
        return lambda true, pred: f1_score(true, pred, average='micro')
    elif name == 'mae':
        return lambda true, pred: torch.mean(torch.abs(true - pred))
    


def get_initial_metric(name):
    if name == 'accuracy':
        return 0
    elif name == 'average_precision':
        return 0
    elif name == 'f1':
        return 0
    elif name == 'mae':
        return float('inf')
