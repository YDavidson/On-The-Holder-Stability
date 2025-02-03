#!/bin/bash

## sort
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model sort_mpnn --num_layers 2   --embed_dim 64 --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --blank_vector_method learnable
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model sort_mpnn --num_layers 2   --embed_dim 128 --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --blank_vector_method learnable
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model sort_mpnn --num_layers 2   --embed_dim 256 --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --blank_vector_method learnable
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model sort_mpnn --num_layers 2   --embed_dim 512 --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --blank_vector_method learnable

## adaptive relu
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model adaptive_relu_mpnn --num_layers 2   --embed_dim 64 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --bias --add_sum
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model adaptive_relu_mpnn --num_layers 2   --embed_dim 128 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --bias --add_sum
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model adaptive_relu_mpnn --num_layers 2   --embed_dim 256 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --bias --add_sum
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model adaptive_relu_mpnn --num_layers 2   --embed_dim 512 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train\
 --bias --add_sum

## relu moments
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation relu  --num_layers 2   --embed_dim 64 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation relu  --num_layers 2   --embed_dim 128 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation relu  --num_layers 2   --embed_dim 256 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation relu  --num_layers 2   --embed_dim 512 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train

## sigmoid moments
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation sigmoid  --num_layers 2   --embed_dim 64 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation sigmoid  --num_layers 2   --embed_dim 128 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation sigmoid  --num_layers 2   --embed_dim 256 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model mlp_moments --aggregate moments --activation sigmoid  --num_layers 2   --embed_dim 512 --combine LinearCombination --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train

##gin
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gin --num_layers 2  --norm batch   --embed_dim 64 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train 
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gin --num_layers 2  --norm batch   --embed_dim 128 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gin --num_layers 2  --norm batch   --embed_dim 256 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gin --num_layers 2  --norm batch   --embed_dim 512 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train

## gat
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gat --num_layers 2 --heads 4 --embed_dim 64 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gat --num_layers 2 --heads 4 --embed_dim 128 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gat --num_layers 2 --heads 4 --embed_dim 256 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gat --num_layers 2 --heads 4 --embed_dim 512 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train

## gcn
# embed dim 64
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gcn --num_layers 2 --embed_dim 64 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 128
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gcn --num_layers 2 --embed_dim 128 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 256
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gcn --num_layers 2 --embed_dim 256 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
# embed dim 512
python train.py --dataset epsilon_trees --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --height 4 --feature_dimension 1 --min_epsilon 0.1 --max_epsilon 1.0 --base_feature 1\
 --model gcn --num_layers 2 --embed_dim 512 --out_mlp_layers 1\
 --lr 0.01 --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 10 --loss ce --metric accuracy --seed 0 --project_name eps_tree_train
 
EOF