from experiments.combine_exp import CombineExperiment
import os





seed = 0
experiment = CombineExperiment()
experiment.run(seed=seed)
if not os.path.exists('exp_plots'):
    os.makedirs('exp_plots')
out_path = os.path.join('exp_plots', 'combine_experiment.png')
experiment.plot(save_path=out_path)
