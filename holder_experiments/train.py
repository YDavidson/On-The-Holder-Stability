import torch
import wandb
import os
import argparse
import json
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from losses import setup_loss
from data_handling import setup_dataloaders
from models import setup_model
from optimizers import setup_optimizer
from metrics import setup_metric
from trainer.trainer import Trainer
from utils import api_keys
from utils.args import get_args
import time
from torch_geometric.seed import seed_everything
from torch.optim import Optimizer
import torch.optim as optim
import math
import random
import numpy as np
from metrics.metric import Metric
from pprint import pprint

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")




def setup_wandb(config, project_name, disable=False, group=None, job_type=None, fold=None, sweep_flag=False):
    run_name = f'{job_type}-{fold}' if fold is not None else job_type
    wandb.login(key=api_keys.WANDB_KEY)
    wandb.init(project=project_name,
               config=config,
               mode='disabled' if disable else 'online',
               group=group,
               allow_val_change=True,
               reinit=True,
               job_type=job_type,
               name=run_name)
    if not sweep_flag:
        wandb.run.name = wandb.run.id
    configs = wandb.config
    summary_path = os.path.join(configs.summary_path, wandb.run.id)
    print('Summary path:', summary_path)
    wandb.config.update({'run_summary_path': summary_path}, allow_val_change=True)
    os.makedirs(summary_path, exist_ok=True)
    return configs


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            print(f"Removing env variable {key}")
            del os.environ[key]


def compute_final_metrics(train_curves, val_curves, val_losses, test_curves):
    train_curves = np.array(train_curves)
    val_curves = np.array(val_curves)
    val_losses = np.array(val_losses)
    test_curves = np.array(test_curves)

    mean_val_losses = np.mean(val_losses, axis=0)
    train_mean_curve = np.mean(train_curves, axis=0)
    train_std_curve = np.std(train_curves, axis=0)
    val_mean_curve = np.mean(val_curves, axis=0)
    val_std_curve = np.std(val_curves, axis=0)
    test_mean_curve = np.mean(test_curves, axis=0)
    test_std_curve = np.std(test_curves, axis=0)

    best_epoch_loss = np.argmin(mean_val_losses)
    loss_best_train_metric_mean = train_mean_curve[best_epoch_loss]
    loss_best_train_metric_std = train_std_curve[best_epoch_loss]
    loss_best_val_metric_mean = val_mean_curve[best_epoch_loss]
    loss_best_val_metric_std = val_std_curve[best_epoch_loss]
    if len(test_mean_curve) > 1:
        loss_best_test_metric_mean = test_mean_curve[best_epoch_loss]
        loss_best_test_metric_std = test_std_curve[best_epoch_loss]
    else:
        loss_best_test_metric_mean = test_mean_curve[0]
        loss_best_test_metric_std = test_std_curve[0]

    best_epoch_metric = Metric.compute_best_metric(args.metric, val_mean_curve, return_index=True)
    metric_best_train_metric_mean = train_mean_curve[best_epoch_metric]
    metric_best_train_metric_std = train_std_curve[best_epoch_metric]
    metric_best_val_metric_mean = val_mean_curve[best_epoch_metric]
    metric_best_val_metric_std = val_std_curve[best_epoch_metric]
    if len(test_mean_curve) > 1:
        metric_best_test_metric_mean = test_mean_curve[best_epoch_metric]
        metric_best_test_metric_std = test_std_curve[best_epoch_metric]
    else:
        metric_best_test_metric_mean = test_mean_curve[0]
        metric_best_test_metric_std = test_std_curve[0]

    
    # return dict of metric means and stds
    return {"loss_best_train_metric_mean": loss_best_train_metric_mean,
            "loss_best_train_metric_std": loss_best_train_metric_std,
            "loss_best_val_metric_mean": loss_best_val_metric_mean,
            "loss_best_val_metric_std": loss_best_val_metric_std,
            "loss_best_test_metric_mean": loss_best_test_metric_mean,
            "loss_best_test_metric_std": loss_best_test_metric_std,
            "metric_best_train_metric_mean": metric_best_train_metric_mean,
            "metric_best_train_metric_std": metric_best_train_metric_std,
            "metric_best_val_metric_mean": metric_best_val_metric_mean,
            "metric_best_val_metric_std": metric_best_val_metric_std,
            "metric_best_test_metric_mean": metric_best_test_metric_mean,
            "metric_best_test_metric_std": metric_best_test_metric_std
            }



def run_train(args, wandb_configs, project_name, no_wandb, sweep_flag=False):
    # set seed
    set_seed(args.seed)
    # seed_everything(args.seed)
    # get data
    train_loaders, val_loaders, test_loaders = setup_dataloaders(args)
    # get in and out dim
    in_dim = train_loaders[0].dataset[0].x.shape[-1]
    out_dim = train_loaders[0].dataset.num_classes
    args.in_dim = in_dim    
    args.out_dim = out_dim
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if sweep_flag or len(train_loaders) > 1 or args.num_seeds > 1:
        if sweep_flag:
            sweep_run = wandb.init()
        else:
            sweep_run = wandb.init(project=project_name)
        sweep_id = sweep_run.sweep_id or "unknown"
        project_url = sweep_run.get_project_url()
        sweep_group_url = f'{project_url}/groups/{sweep_id}'
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
        sweep_run_id = sweep_run.id
        sweep_run.finish()
        wandb.sdk.wandb_setup._setup(_reset=True)
    else:
        sweep_run_name = None
        # generate random id for run
        sweep_run_id = f'{time.time()}_{random.randint(0, 1000)}'
    
    train_curves = []
    val_curves = []
    val_losses = []
    test_curves = []
    assert args.num_seeds == 1 or args.num_folds == 1 or args.num_folds is None, 'Either num_seeds or num_folds should be 1.'
    for seed in range(args.seed, args.seed + args.num_seeds):
        set_seed(seed)    
        for i, (train_loader, val_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
            print(f'Fold {i+1}/{len(train_loaders)}')
            # setup wandb for fold
            reset_wandb_env()
            configs = setup_wandb(wandb_configs, project_name, no_wandb, group=sweep_run_id, 
                                job_type=sweep_run_name, fold=i, sweep_flag=sweep_flag)
            # set all args according to configs
            for key, value in configs.items():
                setattr(args, key, value)
            # dump configs in summary path 
            with open(os.path.join(configs.run_summary_path, 'config.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)  

            # get model
            model = setup_model(args).to(device)        
            # get optimizer
            optimizer = setup_optimizer(model, args)
            # get lr scheduler
            if args.lr_scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            elif args.lr_scheduler == 'step':
                scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.5)
            elif args.lr_scheduler == 'cosine_warmup':
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=args.epochs)
            else:
                scheduler = None
            # get criterion
            criterion = setup_loss(args)
            # get trainer
            trainer = Trainer(model, criterion, args.metric, optimizer, scheduler, train_loader, val_loader, test_loader, 
                            device, args.run_summary_path, args.task_level, args.grad_clip, fold=i)
            # train
            fold_train_curve, fold_val_curve, fold_val_loss, fold_test_curve = trainer.train(args.epochs, args.checkpoint_every)
            # append fold metrics
            train_curves.append(fold_train_curve)
            val_curves.append(fold_val_curve)
            val_losses.append(fold_val_loss)
            test_curves.append(fold_test_curve)
            if not sweep_flag and  len(train_loaders) == 1 and args.num_seeds == 1:
                # log metric to sweep run
                final_metrics = compute_final_metrics(train_curves, val_curves, val_losses, test_curves)
                wandb.log({**final_metrics})
            # finish wandb run
            wandb.finish()
    

    final_metrics = compute_final_metrics(train_curves, val_curves, val_losses, test_curves)
    

    # resume the sweep run
    if sweep_flag or len(train_loaders) > 1 or args.num_seeds > 1:
        if sweep_flag:
            sweep_run = wandb.init(id=sweep_run_id, resume="must")
        else:
            sweep_run = wandb.init(id=sweep_run_id, project=project_name, resume="must")
        # log metric to sweep run
        sweep_run.log({**final_metrics})
        
    pprint(final_metrics)

if __name__ == '__main__':
    args = get_args()

    sweep_id = os.getenv("WANDB_SWEEP_ID")

    if sweep_id is not None:
        # part of sweep
        sweep_flag = True
    else:
        # not part of sweep
        sweep_flag = False
    wandb_configs = args
    project_name = args.project_name
    

    run_train(args, wandb_configs, project_name, args.no_wandb, sweep_flag)
    


    

    


