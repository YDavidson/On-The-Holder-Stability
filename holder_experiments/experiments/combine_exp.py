
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.combine import LinearCombination, LTSum, Concat, ConcatProject



scheme_colors = {
    'LC': 'r',
    'LTSum': 'b',
    'C': 'g',
    'CP': 'y',
}




class CombineExperiment:
    def __init__(self, in_dims=[8, 16, 32, 500], repeat_options=[1, 2, 4, 8, 16, 32, 64, 128], 
                 distortion_quantile=1.0, num_samples=1000, mean=0, std=1):
        self.in_dims = in_dims
        self.repeat_options = repeat_options
        self.distortion_quantile = distortion_quantile
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.results = {}

    def run(self, seed):
        torch.manual_seed(seed)

        for in_dim in self.in_dims:
            samples = [
                (
                    torch.normal(torch.ones(in_dim) * self.mean, self.std),
                    torch.normal(torch.ones(in_dim) * self.mean, self.std),
                    torch.normal(torch.ones(in_dim) * self.mean, self.std),
                    torch.normal(torch.ones(in_dim) * self.mean, self.std),
                )
                for _ in range(self.num_samples)
            ]

            result_dict = {
                'LC': ([], []),
                'LTSum': ([], []),
                'C': ([], []),
                'CP': ([], []),
            }

            # linear combination
            for num_repeats in self.repeat_options:
                kwargs = {'in_dim': in_dim, 'num_repeats': num_repeats}
                distortion, embed_dim = self.run_experiment(samples, LinearCombination, kwargs)
                result_dict['LC'][0].append(embed_dim)
                result_dict['LC'][1].append(distortion)

            # linear transformation & sum
            for num_repeats in self.repeat_options:
                kwargs = {'in_dim_1': in_dim, 'in_dim_2': in_dim, 'num_repeats': num_repeats}
                distortion, embed_dim = self.run_experiment(samples, LTSum, kwargs)
                result_dict['LTSum'][0].append(embed_dim)
                result_dict['LTSum'][1].append(distortion)

            # concat
            for num_repeats in [1]:
                kwargs = {'in_dim_1': in_dim, 'in_dim_2': in_dim}
                distortion, embed_dim = self.run_experiment(samples, Concat, kwargs)
                result_dict['C'][0].append(embed_dim)
                result_dict['C'][1].append(distortion)

            # concat & project
            for num_repeats in [in_dim * rep for rep in self.repeat_options]:
                kwargs = {'in_dim_1': in_dim, 'in_dim_2': in_dim, 'num_repeats': num_repeats}
                distortion, embed_dim = self.run_experiment(samples, ConcatProject, kwargs)
                result_dict['CP'][0].append(embed_dim)
                result_dict['CP'][1].append(distortion)

            self.results[in_dim] = result_dict

    def run_experiment(self, samples, combine_scheme, combine_kwargs):
        combine = combine_scheme(**combine_kwargs)
        orig_dists = []
        embed_dists = []
        metric_ratios = []

        with torch.no_grad():
            for (x, y, z, w) in samples:
                orig_dist = torch.linalg.norm(x - z) + torch.linalg.norm(y - w)
                out_1 = combine(x, y)
                out_2 = combine(z, w)
                embed_dist = torch.linalg.norm(out_1 - out_2)
                orig_dists.append(orig_dist)
                embed_dists.append(embed_dist)
                metric_ratios.append(embed_dist / orig_dist)

        bottom_quantile = (1 - self.distortion_quantile) / 2
        top_quantile = 1 - bottom_quantile

        distortion = np.quantile(metric_ratios, top_quantile) / np.quantile(metric_ratios, bottom_quantile)

        return distortion, combine.embedding_dim

    def plot(self, save_path=None, show_plot=False):
        SMALL_SIZE = 24
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        fig, axs = plt.subplots(1, 4, figsize=(26, 6), constrained_layout=False)
        fig.text(0.5, 0.005, 'embed_dim', ha='center')
        fig.text(0.08, 0.5, f'distortion', va='center', rotation='vertical')

        for i, in_dim in enumerate(self.in_dims):
            ax = axs[i]
            ax.set_title(f'{in_dim=}')

            for scheme, (embed_dims, distortions) in self.results[in_dim].items():
                ax.plot(embed_dims, distortions, '*--', label=scheme, color=scheme_colors[scheme])

        axs[0].legend()

        if save_path is not None:
            plt.savefig(save_path)
        
        if show_plot:
            fig.show()


