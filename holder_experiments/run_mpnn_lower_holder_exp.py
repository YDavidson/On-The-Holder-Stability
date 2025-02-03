import matplotlib.pyplot as plt
import torch
import os
import random
from models.combine import LinearCombination
from data_handling.epsilon_trees import create_epsilon_trees_dataset
from experiments.lower_holder_exp import MPNNLowerHolderExperiment
from torch_geometric.datasets import TUDataset
import argparse
import numpy as np



def run_TUD_experiment(dataset, device, models, seed=0, depths=[3, 7, 11, 15], embed_dim=2048, num_pairs=100, 
                       batch_size=10, combine=LinearCombination):
    # set random seeds
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = TUDataset('data', name=dataset)

    exp = MPNNLowerHolderExperiment(dataset, seed=seed)
    exp.run_experiment(depths, embed_dim, num_pairs, batch_size, combine, device, models=models)

    return exp

def plot_and_save_TUD(exp, scatter_save_path=None, distortion_save_path=None):
    exp.plot_model_scatter(show_plot=False, save_path=scatter_save_path)
    exp.plot_distortion_per_depth(show_plot=False, save_path=distortion_save_path)




def run_epsilon_experiment(tree_heights, depths,  device, models=['sort', 'sigmoid', 'relu', 'adaptive_relu'], 
                           num_pairs=100, feature_dimension=1, min_epsilon=0.05, max_epsilon=0.4, add_random=False, 
                           base_feature=1, embed_dim=45000, combine=LinearCombination,seed=0, batch_size=1, linspace_bias=False):
    # set random seeds
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = []
    for height in tree_heights:
      height_data = create_epsilon_trees_dataset(num_pairs, height, feature_dimension, min_epsilon, max_epsilon, base_feature=base_feature,
                                            directed=False, add_random=add_random)
      dataset.extend(height_data)

    exp = MPNNLowerHolderExperiment(dataset, tree_heights, seed)
    pair_idx = [(2*i, 2*i+1) for i in range(len(dataset)//2)]
    exp.run_experiment(depths, embed_dim, num_pairs, batch_size, combine, device, pair_idx, models, linspace_bias)

    return exp


cmap = {
        0:'r',
        1:'r',
        2:'b',
        3:'b',
        4:'g',
        5:'g',
        6: 'y',
        7: 'y'}
styles = {
    0:'--',
    1:'-.',
    2:':',
    3: '-'
}


    

# EPSILON TREES
def run_all_epsilon_experiments():
    SMALL_SIZE = 18
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 18
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    torch.set_default_dtype(torch.float64)
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    fig.text(0.5, -0.03, 'TMD', ha='center')
    fig.text(-0.03, 0.5, 'embedding distance', va='center', rotation='vertical')
    # create subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)

    # param setup for tree height=3
    tree_heights = [3]
    depths = [1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}')
    
    # test default tensor type
    a = torch.tensor([1.0], device=device)
    print(f'{a.type()=}')

    print('Running experiment on epsilon trees with height=3')
    exp_epsilon_trees_3 = run_epsilon_experiment(tree_heights, depths, device)
    exp_epsilon_trees_3.plot_model_scatter(subfigs=[subfigs[0]], show_plot=False)

    # param setup for tree height=4
    tree_heights = [4]
    depths = [2]

    print('Running experiment on epsilon trees with height=4')
    exp_epsilon_trees_4 = run_epsilon_experiment(tree_heights, depths, device)
    exp_epsilon_trees_4.plot_model_scatter(subfigs=[subfigs[1]], step_size=0.2, show_plot=False)

    # save fig
    if not os.path.exists('exp_plots'):
        os.makedirs('exp_plots')

    fig.savefig(os.path.join('exp_plots', 'epsilon_trees_scatter.png'), dpi=300, bbox_inches='tight')

    # plot empirical vs. theoretical lower-Holder exponents
    fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
    axs[0].set_title('Theoretical exponents')
    axs[1].set_title('Empirical exponents')

    for i, key in enumerate(exp_epsilon_trees_4.theoretical_exponents.keys()):
        axs[0].plot(exp_epsilon_trees_3.depths + exp_epsilon_trees_4.depths,
                    exp_epsilon_trees_3.theoretical_exponents[key] + exp_epsilon_trees_4.theoretical_exponents[key],
                    c=cmap[2*i], label=f'{key}_theoretical', linestyle=styles[i])
        axs[1].plot(exp_epsilon_trees_3.depths + exp_epsilon_trees_4.depths,
                    exp_epsilon_trees_3.empirical_exponents[key] + exp_epsilon_trees_4.empirical_exponents[key],
                    c=cmap[2*i+1], label=f'{key}_empirical', linestyle=styles[i])
    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()

    fig.savefig(os.path.join('exp_plots', 'epsilon_trees_exponents.png'), dpi=300, bbox_inches='tight')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_epsilon', action='store_true') # default is False
    args = parser.parse_args()

    if not args.disable_epsilon:
        run_all_epsilon_experiments()
