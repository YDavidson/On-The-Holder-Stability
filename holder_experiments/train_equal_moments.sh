#!/bin/bash

## sort
# k 3
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 3 --min_epsilon 0.01 --max_epsilon 0.1 --model sort_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --blank_vector_method learnable
#  k 2
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 2 --min_epsilon 0.01 --max_epsilon 0.1 --model sort_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --blank_vector_method learnable
# k 1
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 1 --min_epsilon 0.01 --max_epsilon 0.1 --model sort_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --blank_vector_method learnable
# k 0
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 0 --min_epsilon 0.01 --max_epsilon 0.1 --model sort_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --collapse_method matrix --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --blank_vector_method learnable



## adaptive relu
# k 3
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 3 --min_epsilon 0.01 --max_epsilon 0.1 --model adaptive_relu_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --bias --add_sum

# k 2
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 2 --min_epsilon 0.01 --max_epsilon 0.1 --model adaptive_relu_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --bias --add_sum

# k 1
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 1 --min_epsilon 0.01 --max_epsilon 0.1 --model adaptive_relu_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --bias --add_sum

# k 0
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 0 --min_epsilon 0.01 --max_epsilon 0.1 --model adaptive_relu_mpnn --num_layers 0 --embed_dim 100\
  --combine LinearCombination --out_mlp_layers 1 --lr 0.01 --optimizer adamw\
   --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0 --project_name eq_moments\
    --bias --add_sum


## relu moments
# k 3
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 3 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation relu  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 2
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 2 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation relu  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 1
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 1 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation relu  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 0
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 0 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation relu  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments


## sigmoid moments
# k 3
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 3 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation sigmoid  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 2
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 2 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation sigmoid  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 1
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 1 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation sigmoid  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments

# k 0
python train.py --dataset equal_moments --no_shuffle_train --task classification --task_level graph --batch_size 2\
 --num_pairs 100 --k 0 --min_epsilon 0.01 --max_epsilon 0.1 --model mlp_moments --aggregate moments\
  --activation sigmoid  --num_layers 0 --embed_dim 100 --combine LinearCombination --out_mlp_layers 1 --lr 0.01\
   --optimizer adamw --weight_decay 0.0 --dropout 0.0 --epochs 25 --loss ce --metric accuracy --seed 0\
    --project_name eq_moments


EOF