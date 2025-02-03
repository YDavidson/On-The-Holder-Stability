from metrics import setup_metric
import torch
import numpy as np

class Metric:
    def __init__(self, name):
        self.metric = setup_metric(name)
        self.name = name
        self.gt_list = []
        self.pred_list = []

    def reset(self):
        self.gt_list = []
        self.pred_list = []

    def update(self, y_true, y_pred):
        with torch.no_grad():
            if self.name == 'accuracy':
                y_pred = torch.argmax(y_pred, dim=1)
            elif self.name == 'average_precision':
                y_pred = torch.sigmoid(y_pred)
            elif self.name == 'f1':
                y_pred = torch.argmax(y_pred, dim=1)
            elif self.name == 'mae':
                pass
            
        self.gt_list.append(y_true.detach().cpu())
        self.pred_list.append(y_pred.detach().cpu())

    def better(self, current_best, cadidate_value):
        if self.name == 'accuracy':
            return current_best <= cadidate_value
        elif self.name == 'average_precision':
            return current_best <= cadidate_value
        elif self.name == 'f1':
            return current_best <= cadidate_value
        elif self.name == 'mae':
            return current_best >= cadidate_value
        else:
            raise ValueError(f'Unknown metric {self.name}')
    

    def __call__(self):
        gt = torch.cat(self.gt_list, dim=0)
        pred = torch.cat(self.pred_list, dim=0)
        return self.metric(gt, pred)
        

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
    
    @staticmethod
    def compute_best_metric(metric_name, metric_values, return_index=True):
        metric_values = np.array(metric_values)
        if metric_name in ['accuracy', 'average_precision', 'f1']:
            best_idx = np.argmax(metric_values)
        elif metric_name in ['mae']:
            best_idx = np.argmin(metric_values)
        else:
            raise ValueError(f'Unknown metric {metric_name}')
        if return_index:
            return best_idx
        else:
            return metric_values[best_idx]