import torch
import wandb
import os
from tqdm import tqdm
from metrics import get_initial_metric
from metrics.metric import Metric
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, criterion, metric, optimizer, scheduler, train_loader, val_loader, test_loader, 
                 device, summary_path, task_level='graph', grad_clip=None, fold=0):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.summary_path = summary_path
        self.task_level = task_level
        self.grad_clip = grad_clip
        self.fold = fold
        if task_level not in ['graph', 'node']:
            raise ValueError('task_level must be one of [graph|node]')
        self.return_node_embeddings = (task_level == 'node')
        

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        metric_wrapper = Metric(self.metric)
        num_batches = len(self.train_loader)
        for batch in self.train_loader:
            data = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch, return_node_embeddings=self.return_node_embeddings)
            pred = outputs[2] if self.return_node_embeddings else outputs[0]
            loss = self.criterion(pred, data.y)
            loss.backward()
            if self.grad_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                metric_wrapper.update(data.y, pred)
        if self.scheduler is not None:
            self.scheduler.step()

        result_dict = {"train_loss": total_loss / num_batches}
        result_dict["train_" + self.metric] = metric_wrapper()

        return result_dict

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        metric_wrapper = Metric(self.metric)
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                data = batch.to(self.device)
                outputs = self.model(batch, self.return_node_embeddings)
                pred = outputs[2] if self.return_node_embeddings else outputs[0]
                loss = self.criterion(pred, data.y)
                total_loss += loss.item()
                metric_wrapper.update(data.y, pred)

        average_loss = total_loss / num_batches
        result_dict = {"val_loss": average_loss}
        epoch_metric = metric_wrapper()
        result_dict["val_" + self.metric] = epoch_metric
        

        if metric_wrapper.better(self.best_metric, epoch_metric):
            self.best_metric = epoch_metric
            self.save_model(os.path.join(self.summary_path, "best_model.pth"))
        if average_loss <= self.best_loss:
            self.best_loss = average_loss
            
        return result_dict

    def train(self, num_epochs, checkpoint_every):
        self.best_loss = float('inf')
        self.best_metric = get_initial_metric(self.metric)
        train_metrics = []
        val_metrics = []
        val_losses = []
        test_metrics = []
        for epoch in tqdm(range(num_epochs)):
            train_results = self.train_epoch()
            val_results = self.validate_epoch()
            train_metrics.append(train_results["train_" + self.metric])
            val_metrics.append(val_results["val_" + self.metric])
            val_losses.append(val_results["val_loss"])
            
            if len(self.test_loader) == 0:
                # The case for TU datasets, since we don't have test set    
                test_metrics.append(val_results["val_" + self.metric])
            
            wandb.log({"epoch": epoch + 1,
                    **train_results,
                    **val_results})
            if (epoch + 1) % checkpoint_every == 0:
                self.save_model(os.path.join(self.summary_path, "checkpoint_{}.pth".format(epoch + 1)))

        if len(self.test_loader) > 0:
            test_results = self.test_epoch(self.test_loader)
            test_metrics.append(test_results["test_" + self.metric])

        wandb.log({"best_val_loss": self.best_loss})
        wandb.log({"best_val_"+self.metric: self.best_metric})
        return train_metrics, val_metrics, val_losses, test_metrics

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
    
    def test_epoch(self, test_loader=None):
        self.load_model(os.path.join(self.summary_path, "best_model.pth"))
       
        self.model.eval()
        total_loss = 0.0
        metric_wrapper = Metric(self.metric)
        num_batches = len(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                data = batch.to(self.device)
                outputs = self.model(batch, self.return_node_embeddings)
                pred = outputs[2] if self.return_node_embeddings else outputs[0]
                loss = self.criterion(pred, data.y)
                total_loss += loss.item()
                metric_wrapper.update(data.y, pred)

        average_loss = total_loss / num_batches
        test_metric = metric_wrapper()
        test_results = {"test_loss": average_loss}
        test_results["test_" + self.metric] = test_metric
        return test_results

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

