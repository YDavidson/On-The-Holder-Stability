import torch
from torch_geometric.utils import degree



def get_max_degree_and_nodes(dataset):
  """
  Given a dataset, returns:
  1) Max degree of any node across all dataset.
  2) Max nodes per graph across all graphs in dataset.
  """
  ## get max neighbors and max nodes
  max_degree = 0
  max_nodes = 0
  num_node_features = dataset[0].x.shape[0]

  for datum in dataset:
    max_nodes = max(max_nodes, datum.x.shape[0])
    try:
      max_degree = max(max_degree, torch.max(degree(datum.edge_index[1])).item())
    except:
      max_degree = max(max_degree, 0)

  return int(max_degree), int(max_nodes)

def count_num_params(model):
  """
  Given a model, returns the number of parameters in the model.
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)