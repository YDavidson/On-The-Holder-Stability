
import argparse


def postprocess_args(args):
    if args.grad_clip is not None and args.grad_clip < 0:
        args.grad_clip = None

    return args



def get_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=['mutag', 'proteins', 'ptc', 'nci1', 'nci109',
                                 'epsilon_trees', 'equal_moments'])
    parser.add_argument('--no_shuffle_train', action='store_true')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression', 'multilabel_classification'])
    parser.add_argument('--task_level', type=str, default='graph')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    # positional (node) encoding args
    parser.add_argument('--positional_encoding', default='none',
                        choices=['none', 'atom', 'rwse', 'lape', 'atom+rwse', 'atom+lape'])
    parser.add_argument('--rw_steps', type=int, default=20)
    parser.add_argument('--lape_k', type=int, default=7)
    parser.add_argument('--pe_embed_dim', type=int, default=2) 
    # epsilon trees args
    parser.add_argument('--num_pairs', type=int, default=100)
    parser.add_argument('--height', type=int, default=4)
    parser.add_argument('--feature_dimension', type=int, default=1)
    parser.add_argument('--min_epsilon', type=float, default=0.1)
    parser.add_argument('--max_epsilon', type=float, default=1.0)
    parser.add_argument('--base_feature', type=float, default=1)
    # equal moments args
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--add_outliers', action='store_true')

    ####################################################################################
    # model args
    parser.add_argument('--model', type=str, default='sort_mpnn',
                        choices=['mlp_moments', 'sort_mpnn', 'adaptive_relu_mpnn',
                                 'gin', 'gat', 'gcn'])
    ## General model args
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--combine', type=str, default='LinearCombination',
                        choices=['LinearCombination', 'LTSum', 'Concat', 'ConcatProject'])
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--out_mlp_layers', type=int, default=1)
    parser.add_argument('--out_mlp_dropout', action='store_true') 
    parser.add_argument('--norm', type=str, default=None,
                        choices=[None, 'batch', 'layer'])
    parser.add_argument('--train_out_mlp_only', action='store_true')
    parser.add_argument('--skip_connections', action='store_true')
    ## MLP Moments args
    parser.add_argument('--aggregate', type=str, default='moments')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--bias_ranges', type=float, nargs='+', default=None)
    parser.add_argument('--linspace_bias', action='store_true')
    ## Sort MPNN args
    parser.add_argument('--collapse_method', type=str, default='matrix',
                        choices=['matrix', 'vector'])
    parser.add_argument('--bias', action='store_false')
    parser.add_argument('--update_w_orig', action='store_true')
    parser.add_argument('--blank_vector_method', type=str, default='iterative_update',
                        choices=['iterative_update', 'zero', 'learnable'])
    ## Adaptive ReLU MPNN args
    parser.add_argument('--add_sum', action='store_false')
    parser.add_argument('--clamp_convex_weight', action='store_false')
    ## GAT args
    parser.add_argument('--heads', type=int, default=4)
    ####################################################################################
    # optimizer args
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        choices=['none', 'cosine', 'step', 'cosine_warmup'])
    parser.add_argument('--lr_decay_step', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--grad_clip', type=float, default=None)
    ####################################################################################
    # loss args
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'mae', 'bce_with_logits'])
    
    ####################################################################################
    # metric args
    parser.add_argument('--metric', type=str, default='accuracy',
                        choices=['accuracy', 'average_precision', 'f1', 'mae'])
    ####################################################################################
    # training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--checkpoint_every', type=int, default=300)
    parser.add_argument('--summary_path', type=str, default='summary')
    parser.add_argument('--num_folds', type=int, default=None)
    ####################################################################################
    # wandb args
    parser.add_argument('--project_name', type=str, default='debug')
    parser.add_argument('--sweep_yaml_path', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')

    args = parser.parse_args()

    args = postprocess_args(args)

    return args