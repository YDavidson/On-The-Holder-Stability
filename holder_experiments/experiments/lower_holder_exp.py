import torch
import torch.nn as nn
import numpy as np
import random
import itertools
import gc
from tqdm import tqdm
from torch_geometric.data import Batch
from models.sort_mpnn import SortMPNN
from models.mlp_moments import MLPMomentMPNN
from models.adaptive_relu_mpnn import AdaptiveReluMPNN
from utils.utils import get_max_degree_and_nodes
from utils.exponent_approximation import approximate_lower_exponent, approximate_upper_lipschitz_bound, compute_theoretical_lower_holder_exponent
from utils.TMD_solver import TMD
import matplotlib.pyplot as plt

name_dict = {
        'sort': 'SortMPNN',
        'sigmoid': 'SmoothMPNN',
        'relu': 'ReLUMPNN',
        'adaptive_relu': 'AdaptMPNN'
    }

class MPNNLowerHolderExperiment():
  def __init__(self, dataset, tree_heights=None, seed=0):
    self.dataset = dataset
    self.tree_heights = None if tree_heights is None else np.array(tree_heights) # relevant only for epsilon tree dataset
    self.max_degree, self.max_nodes = get_max_degree_and_nodes(self.dataset)
    self.in_dim = self.dataset[0].x.shape[-1]
    if self.dataset[0].y is None:
      self.out_dim = 1
    else:
      self.out_dim = self.dataset[0].y.shape[0]
    torch.manual_seed(seed)
    random.seed(seed)


  def prepare_samples(self, num_samples, batch_size, pair_idx=None):
    if pair_idx is None:
      self.pair_idx = []
      for i in range(num_samples):
        idx_1 =  random.randint(0, len(self.dataset)-1)
        idx_2 = idx_1
        while idx_2 == idx_1:
          idx_2 = random.randint(0, len(self.dataset)-1)
        self.pair_idx.append((idx_1, idx_2))

    else:
      self.pair_idx = pair_idx

    # batch_pair_idx
    assert(num_samples%batch_size==0)
    self.batches = [[self.pair_idx[j] for j in range(i*batch_size, (i+1)*batch_size)] for i in range(num_samples//batch_size)]
    # per batch indices
    self.batch_first = [2*i for i in range(batch_size)]
    self.batch_second = [2*i+1 for i in range(batch_size)]

  def compute_tmds(self, depths, pair_idx):
    print('Computing TMDs')
    TMD_depth_list = [depth+1 for depth in depths]
    TMDs = [TMD(self.dataset[idx_1], self.dataset[idx_2], w=1, L=TMD_depth_list) for (idx_1, idx_2) in tqdm(pair_idx)]
    return np.array(TMDs).T.tolist()


  def get_initial_kwargs(self, embed_dim, combine):
    sort_kwargs = {
        'in_dim': self.in_dim,
        'embed_dim': embed_dim,
        'out_dim': self.out_dim,
        'bias': False,
        'max_neighbors': self.max_degree,
        'max_nodes': self.max_nodes,
        'combine': combine,
        'collapse_method': 'matrix',
        'update_w_orig': False,
        'out_mlp_layers': 1
    }

    sigmoid_kwargs = {
        'in_dim': self.in_dim,
        'embed_dim': embed_dim,
        'out_dim': self.out_dim,
        'combine': combine,
        'activation': nn.Sigmoid,
        'out_mlp_layers': 1
    }

    relu_kwargs = {
        'in_dim': self.in_dim,
        'embed_dim': embed_dim,
        'out_dim': self.out_dim,
        'combine': combine,
        'out_mlp_layers': 1
    }

    adaptive_relu_kwargs = {
        'in_dim': self.in_dim,
        'embed_dim': embed_dim,
        'out_dim': self.out_dim,
        'combine': combine,
        'add_sum': True, 
        'clamp_convex_weight': True, 
        'linspace_bias': False,
        'out_mlp_layers': 1
    }

    return sort_kwargs, sigmoid_kwargs, relu_kwargs, adaptive_relu_kwargs


  def compute_exponents(self, depths, embed_dim, combine, device, models=['sort','sigmoid','relu', 'adaptive_relu'],
                        linspace_bias=False, p=0.1, normalize=True):

    sort_kwargs, sigmoid_kwargs, relu_kwargs, adaptive_relu_kwargs = self.get_initial_kwargs(embed_dim, combine)

    self.empirical_exponents = {}
    self.theoretical_exponents = {}
    self.embed_dist_per_depth = [{} for _ in range(len(depths))]

    for model_type in ['sort', 'sigmoid', 'relu', 'adaptive_relu']:
      if model_type in models:
        self.empirical_exponents[model_type] = []
        self.theoretical_exponents[model_type] = []
        for d in self.embed_dist_per_depth:
          d[model_type] = []


    for i, depth in enumerate(depths):
      print(f'{depth=}')
      tmds = self.tmds_per_depth[i]
      embed_dists = self.embed_dist_per_depth[i]
      with torch.no_grad():

        # SortMPNN
        if 'sort' in models:
          print('SortMPNN')
          sort_kwargs['num_layers'] = depth
          sort_kwargs['embed_dim'] = min(sort_kwargs['embed_dim'], 2048)
          print(sort_kwargs['embed_dim'])
          model = SortMPNN(**sort_kwargs).to(device)

          for batch_idx in tqdm(self.batches):
            batch_list = []
            batch = Batch.from_data_list([x for x in itertools.chain.from_iterable(itertools.zip_longest([self.dataset[idx_1] for (idx_1,_) in batch_idx],[self.dataset[idx_2] for (_,idx_2) in batch_idx])) if x])
            batch.to(device)
            ## output from models
            outputs, embeddings = model(batch)
            distances = torch.norm(embeddings[self.batch_first] - embeddings[self.batch_second], dim=-1).cpu().detach().tolist()
            embed_dists['sort'].extend(distances)

          del model
          gc.collect()

        # Sigmoid
        if 'sigmoid' in models:
          print('Sigmoid')
          sigmoid_kwargs['num_layers'] = depth
          sigmoid_kwargs['linspace_bias'] = linspace_bias
          model = MLPMomentMPNN(**sigmoid_kwargs).to(device)
          for batch_idx in tqdm(self.batches):
            batch_list = []
            batch = Batch.from_data_list([x for x in itertools.chain.from_iterable(itertools.zip_longest([self.dataset[idx_1] for (idx_1,_) in batch_idx],[self.dataset[idx_2] for (_,idx_2) in batch_idx])) if x])
            batch.to(device)
            outputs, embeddings = model(batch)
            distances = torch.norm(embeddings[self.batch_first] - embeddings[self.batch_second], dim=-1).cpu().detach().tolist()
            embed_dists['sigmoid'].extend(distances)

          del model
          gc.collect()


        # ReLU
        if 'relu' in models:
          print('ReLU')
          relu_kwargs['num_layers'] = depth
          relu_kwargs['linspace_bias'] = linspace_bias
          model = MLPMomentMPNN(**relu_kwargs).to(device)
          for batch_idx in tqdm(self.batches):
            batch_list = []
            batch = Batch.from_data_list([x for x in itertools.chain.from_iterable(itertools.zip_longest([self.dataset[idx_1] for (idx_1,_) in batch_idx],[self.dataset[idx_2] for (_,idx_2) in batch_idx])) if x])
            batch.to(device)
            outputs, embeddings = model(batch)
            distances = torch.norm(embeddings[self.batch_first] - embeddings[self.batch_second], dim=-1).cpu().detach().tolist()
            embed_dists['relu'].extend(distances)

          del model
          gc.collect()

        # adaptive ReLU
        if 'adaptive_relu' in models:
          print('adaptive ReLU')
          adaptive_relu_kwargs['num_layers'] = depth
          adaptive_relu_kwargs['linspace_bias'] = linspace_bias
          model = AdaptiveReluMPNN(**adaptive_relu_kwargs).to(device)
          for batch_idx in tqdm(self.batches):
            batch_list = []
            batch = Batch.from_data_list([x for x in itertools.chain.from_iterable(itertools.zip_longest([self.dataset[idx_1] for (idx_1,_) in batch_idx],[self.dataset[idx_2] for (_,idx_2) in batch_idx])) if x])
            batch.to(device)
            outputs, embeddings = model(batch)
            distances = torch.norm(embeddings[self.batch_first] - embeddings[self.batch_second], dim=-1).cpu().detach().tolist()
            embed_dists['adaptive_relu'].extend(distances)

          del model
          gc.collect()

        if self.tree_heights is not None:
          try:
            max_seperable_height = np.max(self.tree_heights[self.tree_heights <= depth + 2])
          except:
            print(f'{depth=} can\'t separate given tree heights')
            raise NotImplementedError

        for key in embed_dists.keys():
          if self.tree_heights is not None:
            theoretical_exponent = compute_theoretical_lower_holder_exponent(key, max_seperable_height, self.max_degree, self.max_nodes)
          else:
            theoretical_exponent = compute_theoretical_lower_holder_exponent(key, depth, self.max_degree, self.max_nodes)
            theoretical_exponent = min(10, theoretical_exponent)

          self.theoretical_exponents[key].append(theoretical_exponent)

          empirical_exponent = approximate_lower_exponent(x=tmds, y=embed_dists[key], alpha_low=1, alpha_high=2*theoretical_exponent, step_size=0.2, p=p, normalize=normalize)[0]
          self.empirical_exponents[key].append(empirical_exponent)


  def plot_distortion_per_depth(self, show_plot=True, models=None, save_path=None):
    color_dict ={
        'sort': 'b',
        'sigmoid': 'r',
        'relu': 'g',
        'adaptive_relu': 'm',
    }
    models = self.empirical_exponents.keys() if models is None else models
    fig = plt.figure(constrained_layout=True, figsize=(5,5))
    for key in models:
      distortions = []
      for depth in range(len(self.depths)):
        _, c = approximate_lower_exponent(x=self.tmds_per_depth[depth], y=self.embed_dist_per_depth[depth][key], alpha_low=1, alpha_high=1, step_size=1)
        C = approximate_upper_lipschitz_bound(x=self.tmds_per_depth[depth], y=self.embed_dist_per_depth[depth][key])
        distortions.append(C/c)

      plt.plot(self.depths, distortions, label=f'{key}', color=color_dict[key])

    plt.xlabel('depth')
    plt.ylabel('distortion')
    plt.legend()

    if save_path is not None:
      fig.savefig(save_path, dpi=300, bbox_inches='tight')    
    if not show_plot:
      plt.close()
    



  def plot_model_scatter(self, subfigs=None, show_plot=True, p=0.1, step_size=0.25, add_theoretical_bound=True,
                         save_path=None):
    if subfigs == None:
      fig = plt.figure(constrained_layout=True, figsize=(3*len(self.theoretical_exponents.keys()), 3*len(self.depths)))
      fig.suptitle('TMD vs. embedding distance')
      # set x and y axis labels
      fig.text(0.5, 0.04, 'TMD', ha='center')
      fig.text(0.04, 0.5, r'$L_2$', va='center', rotation='vertical')
      # create subfigs
      subfigs = fig.subfigures(nrows=len(self.depths), ncols=1)
      if len(self.depths) == 1:
        subfigs = [subfigs]
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(f'depth={self.depths[i]}')
        # create subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=len(self.theoretical_exponents.keys()), sharex=True)
        if len(self.theoretical_exponents.keys()) == 1:
          axs=[axs]
        for j, key in enumerate(self.theoretical_exponents.keys()):
            ax = axs[j]
            ax.set_title(f'{name_dict[key]}')

            ax.scatter(self.tmds_per_depth[i], self.embed_dist_per_depth[i][key], c='orange', alpha=1, marker='x')
            max_tmd = np.max(self.tmds_per_depth[i])
            samples=np.linspace(0,max_tmd,1000)
            if add_theoretical_bound:
              alpha = self.theoretical_exponents[key][i]
              real_alpha, real_c = approximate_lower_exponent(x=self.tmds_per_depth[i], y=self.embed_dist_per_depth[i][key],
                                                              alpha_low=alpha, alpha_high=alpha, step_size=1, p=p)
              assert real_alpha == alpha, f'{real_alpha=}, {alpha=}, depth={self.depths[i]}, {key=}'
              label = r'$\alpha$'+'={:.2f}'.format(real_alpha) if key == 'relu' else r'$\alpha$'+'={:.0f}'.format(real_alpha)
              ax.plot(samples,real_c*np.power(samples,real_alpha), label=label)
            ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if not show_plot:
      plt.close()



  def run_experiment(self, depths, embed_dim, num_samples, batch_size, combine, device, pair_idx=None,
                     models=['sort', 'sigmoid', 'relu', 'adaptive_relu'], linspace_bias=False,
                     tmds=None):
    """
    :param: <list of int> depths - the depths of the MPNNs to run experiment for
    :param: <int> embed_dim - embedding dimension of the MPNNs
    :param: <int> num_samples - number of pairs for which to compute TMD vs. embedding distance
    :param:<int> batch_size - batch size for running samples. Must divide num_samples
    :param: combine - combine scheme to use in MPNN Combine
    :param: device - cuda or cpu
    :param: <list of tuples (idx_1,idx_2)> pair_idx - pairs of matching indices for TMD and embedding distance computation.
    :param: <list of str> models - names of models to use in experiment. values must be from [sort|sigmoid|relu|adaptive_relu]
    :param: <bool> linspace_bias - wether to initialize biases in [sigmoid|relu|adaptive_relu] with equally spaced values
    """
    self.embed_dim = embed_dim
    self.combine = combine
    self.depths = depths
    self.device = device

    self.prepare_samples(num_samples, batch_size, pair_idx)
    if tmds is None:
        self.tmds_per_depth = self.compute_tmds(depths, self.pair_idx)
    else:
        self.tmds_per_depth = tmds
    self.compute_exponents(depths, embed_dim, combine, device, models, linspace_bias)

