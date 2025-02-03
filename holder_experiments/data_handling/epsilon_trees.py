import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import files_exist
import numpy as np
import os
from typing import Sequence, Type


def create_constant_quad_tree(height, feature_vector, directed):
  """
  :param: height - height of the quad tree
  :param: feature vector - the feature to use for all nodes in the graph
  :param: directed - whether the trees should be directed

  :return: node features
  :return: edges
  """
  assert height > 1
  # create edges
  edges = torch.tensor([[],[]], dtype=torch.long)
  parents = [0]
  start = 1
  num_leaves = 1
  for i in range(1, height):
    stop = start + 4**i
    children = [*range(start, stop)]
    src_edges = [parents[i//4] for i in range(len(parents)*4)]
    curr_height_edges = torch.tensor([children,
                                     src_edges], dtype=torch.long)
    edges = torch.cat([edges, curr_height_edges], dim=-1)
    start = stop
    parents = children
    num_leaves = len(children)

  # make symmetric
  if not directed:
    edges = torch.cat([edges.flip(dims=[0]), edges], dim=-1)

  # create node features
  if len(feature_vector.shape) == 1:
    feature_vector = feature_vector.unsqueeze(0)

  node_features = feature_vector.repeat(int((4**height - 1)/(3)), 1)

  return node_features, edges, num_leaves


def get_leaf_features(height, eps):
  a = [eps, -eps, eps, -eps]
  b = [2*eps, -2*eps, 2*eps, -2*eps]
  c = [eps, -eps, 2*eps, -2*eps]
  d = [eps, -eps, 2*eps, -2*eps]
  
  for i in range(height-3):
    a_prev = a
    b_prev = b
    c_prev = c
    d_prev = d

    a = a_prev + b_prev + a_prev + b_prev
    b = c_prev + d_prev + c_prev + d_prev
    c = a_prev + b_prev + c_prev + d_prev
    d = a_prev + b_prev + c_prev + d_prev

  return torch.tensor((a + b)).unsqueeze(0).T, torch.tensor(c + d).unsqueeze(0).T





def create_epsilon_trees_dataset(num_pairs, height, feature_dimension, min_epsilon, max_epsilon, 
                                 base_feature=1, directed=False, add_random=True, task='classification'):
  """
  :param: num_pairs - number of pairs to have in the dataset
  :param: depth - depths of trees
  :param: feature_dimension - dimension of node features
  :param: min_epsilon - minimal value of epsilon for feature addition
  :param: max_epsilon - maximal value of epsilon for feature addition
  :param: directed - whether the trees should be directed

  :return: dataset - a dataset of graph where graphs [2*i, 2*i+1] are tree pairs as described in the paper.
  """
  assert height > 1, "must use height greater than 1"
  assert feature_dimension == 1, "For now, must use feature dimension 1"

  feature_vector = torch.zeros((feature_dimension))
  left_node_features, left_edges, num_leaves = create_constant_quad_tree(height-1, feature_vector, directed)

  # clone twice
  right_node_features = torch.clone(left_node_features)
  right_edges = left_edges + left_node_features.shape[0]

  # add root
  root_idx = torch.max(right_edges).item() + 1
  root_feature = feature_vector.unsqueeze(0)
  root_edges = torch.tensor([[0, left_node_features.shape[0]],[root_idx, root_idx]])
  if not directed:
    root_edges = torch.cat([root_edges.flip(dims=[0]), root_edges], dim=-1)
  edges = torch.cat([left_edges, right_edges, root_edges], dim=-1)

  dataset = []
  for eps in np.linspace(min_epsilon, max_epsilon, num_pairs):

    leaf_features_1, leaf_features_2 = get_leaf_features(height, eps)
    left_node_features[-num_leaves:] = leaf_features_1[:num_leaves]
    right_node_features[-num_leaves:] = leaf_features_1[num_leaves:]
    features_1 = torch.cat([left_node_features, right_node_features, root_feature], dim=0)
    features_1 += torch.ones_like(features_1)*base_feature
    graph_1 = Data(x=features_1, edge_index=edges, y=torch.tensor([0]) if task == 'classification' else torch.tensor([eps]))
    dataset.append(graph_1)

    left_node_features[-num_leaves:] = leaf_features_2[:num_leaves]
    right_node_features[-num_leaves:] = leaf_features_2[num_leaves:]
    features_2 = torch.cat([left_node_features, right_node_features, root_feature], dim=0)
    features_2 += torch.ones_like(features_2)*base_feature
    graph_2 = Data(x=features_2, edge_index=edges, y=torch.tensor([1]) if task == 'classification' else torch.tensor([-eps]))
    dataset.append(graph_2)

  return dataset



class EpsilonTrees(InMemoryDataset):
  def __init__(self, root, num_pairs, height, feature_dimension, 
               min_epsilon, max_epsilon, base_feature=1, directed=False, 
               add_random=True, transform=None, pre_transform=None,
               task='classification', force_reload=True):
    self.num_pairs = num_pairs
    self.height = height
    self.feature_dimension = feature_dimension
    self.min_epsilon = min_epsilon
    self.max_epsilon = max_epsilon
    self.base_feature = base_feature
    self.directed = directed
    self.add_random = add_random
    assert task in ['classification', 'regression'], "task must be one of ['classification', 'regression']"
    self.task = task
    if force_reload and files_exist([os.path.join(root, 'processed', f'processed_data_h_{self.height}.pt')]):
      # if data exists, delete it
      os.remove(os.path.join(root, 'processed', f'processed_data_h_{self.height}.pt'))


    super(EpsilonTrees, self).__init__(root, transform, pre_transform)
    self.load(self.processed_paths[0])

  @property
  def num_classes(self) -> int:
      return 2 if self.task == 'classification' else 1

  @property
  def raw_file_names(self):
    return []

  @property
  def processed_file_names(self):
    return [f'processed_data_h_{self.height}.pt']

  def download(self):
    pass

  def process(self):
    data_list = create_epsilon_trees_dataset(self.num_pairs, self.height, self.feature_dimension, 
                                             self.min_epsilon, self.max_epsilon, self.base_feature, self.directed, 
                                             self.add_random, self.task)

    self.save(data_list, self.processed_paths[0])

  @classmethod
  def save(cls, data_list: Sequence[BaseData], path: str) -> None:
    r"""Saves a list of data objects to the file path :obj:`path`."""
    data, slices = cls.collate(data_list)
    torch.save((data.to_dict(), slices, data.__class__), path)

  def load(self, path: str, data_cls: Type[BaseData] = Data):
    r"""Loads the dataset from the file path :obj:`path`."""
    data, self.slices, _ = torch.load(path)
    if isinstance(data, dict):  # Backward compatibility.
        data = data_cls.from_dict(data)
    self.data = data






# test
if __name__ == '__main__':
  dataset = EpsilonTrees(root=os.path.join('DATA','epsilon_trees'), num_pairs=5, height=4, feature_dimension=1, min_epsilon=0.1, max_epsilon=1, 
                         base_feature=1, directed=True, add_random=True, task='regression')
  for i in range(len(dataset)):
    print(dataset[i].y)