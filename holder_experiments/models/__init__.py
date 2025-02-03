from models.mlp_moments import MLPMomentMPNN
from models.sort_mpnn import SortMPNN
from models.adaptive_relu_mpnn import AdaptiveReluMPNN
import torch.nn as nn
from models.combine import COMBINE_DICT
from models.gin import GINWrapper
from models.gcn import GCNWrapper
from models.gat import GATWrapper
from utils.utils import count_num_params

activation_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}



def setup_model(args, infer_embedding_dim=False):
    # infer embedding dim for pep dataset where we have a limit on the number of parameters
    if 'pep' in args.dataset and not infer_embedding_dim:
        possible_embed_dims = list(range(110, 230, 10))
        for i, embed_dim in enumerate(possible_embed_dims):
            args.embed_dim = embed_dim
            model = setup_model(args, infer_embedding_dim=True)
            num_params = count_num_params(model)
            if num_params > 500000 or i == len(possible_embed_dims)-1:
                if i == 0:
                    raise ValueError('Embed dim too high for dataset')
                args.embed_dim = possible_embed_dims[i-1]
                print(f'using embed_dim={args.embed_dim}. num_params={num_params}')
                break
            
    if args.model == 'mlp_moments':
        model_kwargs = {k: vars(args)[k] for k in ('in_dim', 'embed_dim', 'out_dim', 'num_layers', 
                                             'combine', 'activation', 'bias_ranges', 'linspace_bias', 'dropout', 'out_mlp_layers',
                                             'norm', 'skip_connections')}
        model_kwargs['activation'] = activation_dict[model_kwargs['activation']]
        model_kwargs['combine'] = COMBINE_DICT[model_kwargs['combine']]
        model = MLPMomentMPNN(**model_kwargs)
    
    elif args.model == 'sort_mpnn':
        model_kwargs = {k: vars(args)[k] for k in ('in_dim', 'embed_dim', 'out_dim', 'num_layers', 'bias',
                                             'max_neighbors', 'max_nodes', 'combine', 'collapse_method', 'update_w_orig', 
                                             'dropout', 'out_mlp_layers', 'out_mlp_dropout',
                                             'norm', 'skip_connections', 'positional_encoding',
                                             'blank_vector_method')}
        model_kwargs['combine'] = COMBINE_DICT[model_kwargs['combine']]
        model_kwargs['args'] = args
        model = SortMPNN(**model_kwargs)
    
    elif args.model == 'adaptive_relu_mpnn':
        model_kwargs = {k: vars(args)[k] for k in ('in_dim', 'embed_dim', 'out_dim', 'num_layers', 
                                             'combine', 'add_sum', 'clamp_convex_weight', 'linspace_bias', 'dropout', 'out_mlp_layers',
                                             'norm', 'skip_connections', 'positional_encoding')}
        model_kwargs['combine'] = COMBINE_DICT[model_kwargs['combine']]
        model_kwargs['args'] = args
        model = AdaptiveReluMPNN(**model_kwargs)
    
    elif args.model == 'gin':
        model = GINWrapper(in_dim=args.in_dim, embed_dim=args.embed_dim, out_dim=args.out_dim, 
                   num_layers=args.num_layers, dropout=args.dropout, jk='cat', norm=args.norm)
    elif args.model == 'gcn':
        model = GCNWrapper(in_dim=args.in_dim, embed_dim=args.embed_dim, out_dim=args.out_dim, 
                   num_layers=args.num_layers, dropout=args.dropout, norm=args.norm)
    elif args.model == 'gat':
        model = GATWrapper(in_dim=args.in_dim, embed_dim=args.embed_dim, out_dim=args.out_dim, 
                   num_layers=args.num_layers, dropout=args.dropout, norm=args.norm, heads=args.heads)
    
    else:
        raise ValueError('Unknown model: {}, must be one of {}'.format(args.model, 
                                                                       ['mlp_moments', 'sort_mpnn', 'adaptive_relu_mpnn',
                                                                        'gin', 'gcn', 'gat']))
    
    if args.train_out_mlp_only:
        # freeze all layers
        for name, param in model.named_parameters():
            if 'out_MLP' not in name:
                param.requires_grad = False

    return model