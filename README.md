**This repository contains the code for the paper *On the Hölder Stability of Multiset and Graph Neural Networks* [ICLR 2025]**


# Setting up the environment

Create the conda environment and activate it.

```bash
conda env create --file=environment.yml
conda activate HolderAnalysis
```

Install pip dependencies

  -  download pytorch version 2.2.2 using the appropriate command for your operating system and CUDA version.
  The relevant command can be found in https://pytorch.org/get-started/previous-versions/
  
  For example, for linux with CUDA 12.1, the command is:
  ```bash
  pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
  ```

  - install ogb version 1.3.6:
  ```bash
  pip install ogb==1.3.6
 ```

  - install performer-pytorch version 1.1.4:
  ```bash
  pip install performer-pytorch==1.1.4
  ```

  - install pot version 0.9.3:
  ```bash
  pip install pot==0.9.3
  ```

  -  install pytorch-lightning version 2.2.2:
  ```bash
  pip install pytorch-lightning==2.2.2
  ```

  - install scipy version 1.13.0:
  ```bash
  pip install scipy==1.13.0
  ```

  - install tensorboardx version 2.6.2.2:
  ```bash
  pip install tensorboardx==2.6.2.2
  ```

  - install torch-geometric version 2.3.0 and pyg-lib (if in linux), torch-cluster, torch-scatter, torch-sparse, torch-spline-conv for pytorch version 2.2.
  
  use the appropriate command for your operating system and CUDA version from https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  
  For example, for linux with CUDA 12.1, the commands are:
  ```bash
  pip install torch-geometric==2.3.0
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
  ```

  - install torchmetrics version 1.3.2:
  ```bash
  pip install torchmetrics==1.3.2 
  ```
  
  - install wandb version 0.16.6:
  ```bash
  pip install wandb==0.16.6
  ```

  - install yacs version 0.1.8:
  ```bash
  pip install yacs==0.1.8 
  ```

  - install matplotlib version 3.8.4:
  ```bash 
  pip install matplotlib==3.8.4
  ```



# holder_experiments

This directory includes the code for the following experiments:
- Multiset lower holder experiment on adversarial example.
- MPNN lower holder experiment on adversarial epsilon-tree example.
- Distortion experiment for COMBINE functions (from appendix).
- Epsilon tree training experiment
- TUDataset experiments. 

For any of the following experiments, first change the directory:
```bash
cd holder_experiments
```

## Setting up wandb key

Update information in `utils/api_keys.py` to include the relevant information

## Running lower Holder experiments
For multisets:
```bash
python run_multiset_lower_holder_exp.py
```
This will produce: 

(1) The figure showing separation quality of ReLU vs. smooth on a single $\pm\epsilon$ pair as a function of width.

(2)  The figure that shows the lower-Holder exponents of the various multiset embeddings on the $\pm\epsilon$ dataset.

For mpnns:
```bash
python run_mpnn_lower_holder_exp.py
```
Result figures will appear under the `exp_plots` directory.


## Running COMBINE experiment
```bash
python run_combine_exp.py
```
Result figure will appear under the `exp_plots` directory.

## Running trained epsilon tree experiments
Run bash file train_eps_tree.sh.
```bash
./train_eps_tree.sh
```
Results can be found in wandb project `eps_tree_train` under `metric_best_test_metric_mean`.

## Running TUDataset experiments
For any choice of $DATASET={mutag, nci1, nci109, ptc, proteins} and $MODEL={adapt, sort}, run the experiment
```bash
./experiment_scripts/$DATASET$_$MODEL$.sh
```

Results can be found in wandb as `metric_best_test_metric_mean` and `metric_best_test_metric_std`.






# lrgb_holder
Based on the GPS codebase: https://github.com/rampasek/GraphGPS. 
This directory includes the peptides-struct and peptides-func experiments

For any of the following experiments, first change the directory:
```bash
cd lrgb_holder
```


## Running lrgb experiments
### Regular experiments
For the results on both peptides-struct and peptides-func with the 500K parameter budget, run for $DATASET={peptides-struct, peptides-func} $MODEL={adaptive-relu, sort}
```bash
python main.py --cfg configs/LRGB-tuned/$DATASET$-$MODEL$.yaml --repeat 4 wandb.use False
```

Results will appear in `results/$DATASET-$MODEL/agg/test/best.json`
### Small models
For the results using small models on peptides-struct, for $MODEL={adaptive-relu, sort, GCN}, $BUDGET={100K, 50K, 25K, 7K}, run
```bash
python main.py --cfg configs/LRGB-tuned/peptides-struct-$MODEL$.yaml --repeat 4 name_tag $BUDGET$ wandb.use False gnn.dim_inner $DIM$
```
Where $DIM$ is the hidden dimension as specified in the relevant table in the appendix. 

To get the graph from the paper, fill the result MAEs in `plot_pep_struct_size_vs_mae.py` and run the script.



# ESAN_holder

Based strongly on repository containing the official code of the paper
**[Equivariant Subgraph Aggregation Networks](https://arxiv.org/abs/2110.02910) (ICLR 2022 Spotlight)**:
https://github.com/beabevi/ESAN/tree/master. 

This code is used for the experiments on ZINC12K using SortMPNN and AdaptMPNN as backbone networks for ESAN.


For any of the following experiments, first change the directory:
```bash
cd ESAN_holder
```

## Prepare the data
Run
```bash
python data.py --dataset ZINC
```

## Run the models

To perform hyperparameter tuning, make use of `wandb`:

1. In `configs/determenistic` folder, choose the `yaml` file corresponding to the dataset and model of interest, say `<config-name>`. This file contains the hyperparameters grid. (Four options: DS-ZINC-sort, DSS-ZINC-sort, DS-ZINC-adaptive_relu, DSS-ZINC-adaptive_relu)

2. Run
    ```bash
    wandb sweep configs/<config-name>
    ````
    to obtain a sweep id `<sweep-id>`. Note the sweep is used to run the same model+hyperparameters just with 10 different seeds.

3. Run the experiment with
    ```bash
    wandb agent <sweep-id>
    ```
    You can run the above command multiple times on each machine you would like to contribute to the grid-search

4. Open your project in your wandb account on the browser to see the results:
    
    compute mean and std of `Metric/test_mean` over the different sweep runs.
