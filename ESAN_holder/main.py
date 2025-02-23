import argparse
import multiprocessing as mp
import os
import random

import numpy as np
import torch
import torch.optim as optim
import wandb
from ogb.graphproppred import Evaluator
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
# noinspection PyUnresolvedReferences
from data import SubgraphData
from utils import get_data, get_model, SimpleEvaluator, NonBinaryEvaluator, Evaluator, get_max_degree_and_nodes

torch.set_num_threads(1)

def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(f'no grad for {name}')
            raise AttributeError(" ‘NoneType’ object has no attribute ‘data’")
    total_norm = total_norm ** (1. / 2)
    return total_norm



def train(model, device, loader, optimizer, criterion, epoch, fold_idx, grad_clip=None):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            y = batch.y.view(pred.shape).to(torch.float32) if pred.size(-1) == 1 else batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])

            wandb.log({f'Loss/train': loss.item()})
            loss.backward()
            grad_norm = get_grad_norm(model)
            wandb.log({f'Loss/grad_norm': grad_norm})
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()


def eval(model, device, loader, evaluator, voting_times=1):
    model.eval()

    all_y_pred = []
    for i in range(voting_times):
        y_true = []
        y_pred = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                y = batch.y.view(pred.shape) if pred.size(-1) == 1 else batch.y
                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

        all_y_pred.append(torch.cat(y_pred, dim=0).unsqueeze(-1).numpy())

    y_true = torch.cat(y_true, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def run(args, device, fold_idx, sweep_run_name, sweep_id, results_queue):
    set_seed(args.seed)

    reset_wandb_env()
    run_name = "{}-{}".format(sweep_run_name, fold_idx)
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=args,
    )

    train_loader, train_loader_eval, valid_loader, test_loader, attributes = get_data(args, fold_idx)
    if args.dataset == 'ZINC':
        args.max_degree, args.max_nodes = get_max_degree_and_nodes(train_loader.dataset + train_loader_eval.dataset + valid_loader.dataset + test_loader.dataset)
    in_dim, out_dim, task_type, eval_metric = attributes

    if 'ogb' in args.dataset:
        evaluator = Evaluator(args.dataset)
    else:
        evaluator = SimpleEvaluator(task_type) if args.dataset != "IMDB-MULTI" \
                                                  and args.dataset != "CSL" else NonBinaryEvaluator(out_dim)

    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if 'ZINC' in args.dataset:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    elif 'ogb' in args.dataset:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    if "classification" in task_type:
        criterion = torch.nn.BCEWithLogitsLoss() if args.dataset != "IMDB-MULTI" \
                                                    and args.dataset != "CSL" else torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.L1Loss()

    # If sampling, perform majority voting on the outputs of 5 independent samples
    voting_times = 5 if args.fraction != 1. else 1

    train_curve = []
    valid_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):

        train(model, device, train_loader, optimizer, criterion, epoch=epoch, fold_idx=fold_idx, grad_clip=args.grad_clip)

        # Only valid_perf is used for TUD
        train_perf = eval(model, device, train_loader_eval, evaluator, voting_times) \
            if 'ogb' in args.dataset else {eval_metric: 300.}
        valid_perf = eval(model, device, valid_loader, evaluator, voting_times)
        test_perf = eval(model, device, test_loader, evaluator, voting_times) \
            if 'ogb' in args.dataset or 'ZINC' in args.dataset else {eval_metric: 300.}

        if scheduler is not None:
            if 'ZINC' in args.dataset:
                scheduler.step(valid_perf[eval_metric])
                if optimizer.param_groups[0]['lr'] < 0.00001:
                    break
            else:
                scheduler.step()

        train_curve.append(train_perf[eval_metric])
        valid_curve.append(valid_perf[eval_metric])
        test_curve.append(test_perf[eval_metric])

        run.log(
            {
                f'Metric/train': train_perf[eval_metric],
                f'Metric/valid': valid_perf[eval_metric],
                f'Metric/test': test_perf[eval_metric]
            }
        )

    wandb.join()

    results_queue.put((train_curve, valid_curve, test_curve))
    return



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




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str,
                        help='Type of convolution {gin, originalgin, zincgin, graphconv, sort, adaptive_relu}')
    parser.add_argument('--random_ratio', type=float, default=0.,
                        help='Number of random features, > 0 only for RNI')
    parser.add_argument('--model', type=str,
                        help='Type of model {deepsets, dss, gnn}')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--channels', type=str,
                        help='String with dimension of each DS layer, separated by "-"'
                             '(considered only if args.model is deepsets)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--jk', type=str, default="last",
                        help='JK strategy, either last or concat (default: last)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training (default: 0.01)')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)')
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--policy', type=str, default="edge_deleted",
                        help='Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}'
                             ' (default: edge_deleted)')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='Depth of the ego net if policy is ego_nets (default: 2)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience (default: 20)')
    parser.add_argument('--test', action='store_true',
                        help='quick test')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    
    parser.add_argument('--combine', type=str, default='LinearCombination',
                        choices=['LinearCombination', 'LTSum', 'Concat', 'ConcatProject'])
    # sort args
    parser.add_argument('--collapse_method', type=str, default='matrix',
                        choices=['matrix', 'vector'])
    parser.add_argument('--bias', action='store_true') # default is False
    parser.add_argument('--blank_method', type=str, default='learnable',
                        choices=['zero', 'learnable'])
    parser.add_argument('--relu_edge', action='store_true',
                        help='whether to add relu following summation of node and edge features') # default is False
    # Adaptive ReLU args
    parser.add_argument('--add_sum', action='store_false') # default is True
    parser.add_argument('--clamp_convex_weight', action='store_false') # default is True
    parser.add_argument('--linspace_bias', action='store_true') # default is False
    # pooling args
    parser.add_argument('--custom_readout', action='store_true') # default is False
    
    
    args = parser.parse_args()
    print(args)
    args.channels = list(map(int, args.channels.split("-")))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed)

    mp.set_start_method('spawn')

    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    if 'ogb' in args.dataset or 'ZINC' in args.dataset:
        n_folds = 1
    elif 'CSL' in args.dataset:
        n_folds = 5
    else:
        n_folds = 10

    # number of processes to run in parallel
    if n_folds > 1 and 'REDDIT' not in args.dataset:
        if args.dataset == 'PROTEINS':
            num_proc = 2
        else:
            num_proc = 3 if args.batch_size == 128 and args.dataset != 'MUTAG' and args.dataset != 'PTC' else 5
    else:
        num_proc = 1

    if args.dataset in ['CEXP', 'EXP']:
        num_proc = 2
    if 'IMDB' in args.dataset and args.policy == 'edge_deleted':
        num_proc = 1

    num_free = num_proc
    results_queue = mp.Queue()

    curve_folds = []
    fold_idx = 0

    if args.test:
        run(args, device, fold_idx, sweep_run_name, sweep_id, results_queue)
        exit()

    while len(curve_folds) < n_folds:
        if num_free > 0 and fold_idx < n_folds:
            p = mp.Process(
                target=run, args=(args, device, fold_idx, sweep_run_name, sweep_id, results_queue)
            )
            fold_idx += 1
            num_free -= 1
            p.start()
        else:
            curve_folds.append(results_queue.get())
            num_free += 1

    train_curve_folds = np.array([l[0] for l in curve_folds])
    valid_curve_folds = np.array([l[1] for l in curve_folds])
    test_curve_folds = np.array([l[2] for l in curve_folds])

    # compute aggregated curves across folds
    train_curve = np.mean(train_curve_folds, 0)
    train_accs_std = np.std(train_curve_folds, 0)

    valid_curve = np.mean(valid_curve_folds, 0)
    valid_accs_std = np.std(valid_curve_folds, 0)

    test_curve = np.mean(test_curve_folds, 0)
    test_accs_std = np.std(test_curve_folds, 0)

    task_type = 'classification' if args.dataset != 'ZINC' else 'regression'
    if 'classification' in task_type:
        best_val_epoch = np.argmax(valid_curve)
        best_train = max(train_curve)
    else:
        best_val_epoch = len(valid_curve) - 1
        best_train = min(train_curve)

    sweep_run.summary[f'Metric/train_mean'] = train_curve[best_val_epoch]
    sweep_run.summary[f'Metric/valid_mean'] = valid_curve[best_val_epoch]
    sweep_run.summary[f'Metric/test_mean'] = test_curve[best_val_epoch]
    sweep_run.summary[f'Metric/train_std'] = train_accs_std[best_val_epoch]
    sweep_run.summary[f'Metric/valid_std'] = valid_accs_std[best_val_epoch]
    sweep_run.summary[f'Metric/test_std'] = test_accs_std[best_val_epoch]

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Val std': valid_accs_std[best_val_epoch],
                    'Test': test_curve[best_val_epoch], 'Test std': test_accs_std[best_val_epoch],
                    'Train': train_curve[best_val_epoch], 'Train std': train_accs_std[best_val_epoch],
                    'BestTrain': best_train}, args.filename)

    wandb.join()


if __name__ == "__main__":
    main()
