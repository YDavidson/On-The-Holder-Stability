import torch
from torch_geometric.utils import to_dense_batch
import torch.nn as nn
from conv import ZINCSortConv, ZINCAdaptiveReLUConv


class ZINCSortGlobalConv(ZINCSortConv):
    def __init__(self, in_dim, out_dim, max_nodes, bias=False, collapse_method='vector',
                 blank_method='learnable'):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param max_nodes: max number of vertices in graph across all dataset
        """
        super().__init__(in_dim, out_dim, max_nodes, bias, collapse_method=collapse_method, 
                         blank_method=blank_method, combine=None)
        self.max_nodes = self.max_degree
        
    def setup_projection(self, in_dim, out_dim, bias):
       return torch.nn.Linear(in_dim, out_dim, bias=bias)

    def setup_bond_encoder(self, in_dim):
       return None
    
    def convert_to_dense(self, inputs, index, indices, num_indices):
      return to_dense_batch(inputs[indices], index[indices], max_num_nodes=self.max_nodes)

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1) # [Num_nodes("batch"), num_neighbors("max_nodes"), out_dim] --> [Num_nodes("batch"), out_dim]
      elif self.collapse_method == 'matrix':
        out = sorted * self.lin_collapse
        out = torch.sum(out, dim=1).squeeze(1)
      else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")
    
      return out
    
    def replace_indices_(self, index):
      with torch.no_grad():
        replacements = torch.stack([self.first_node_index, torch.argsort(self.first_node_index)]).T
        mask = (index == replacements[:, :1])
        index = (1 - mask.sum(dim=0)) * index + (mask * replacements[:,1:]).sum(dim=0)
      return index

    def message(self, x_j):
       return self.lin_project(x_j)

    def update(self, aggr_out, index, batch):
        # return the aggregated out
        return aggr_out
    

    def forward(self, x, batch=None):
        # create batch numbering for single graph if needed
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
          
        self.num_indices = batch.max().item() + 1

        # num nodes per graph in batch
        num_nodes = (torch.unique(batch, return_counts=True)[1]).to(x.device)
        # index of first node per graph
        first_node_index = num_nodes.cumsum(dim=0)
        self.first_node_index = torch.cat([torch.tensor([0], device=x.device), first_node_index], dim=0)[:-1]
        self.max_index = first_node_index[-1].item()
        # setting edge index so all nodes are connected to the first node per graph
        src = torch.arange(x.shape[0], device=x.device)
        dst = torch.cat([torch.ones(num_nodes[i], device=x.device)*self.first_node_index[i] for i in range(0, len(num_nodes))], dim=0)
        edge_index = torch.stack([src, dst], dim=0).long().to(x.device)

        x = self.propagate(edge_index, x=x, blank_vec=self.blank_vec, batch=batch)
        return x
    




class ZincAdaptiveReLUGlobalConv(ZINCAdaptiveReLUConv):
  def __init__(self, in_dim, out_dim, combine, add_sum=True, clamp_convex_weight=True, linspace_bias=False, bias=False):
    super().__init__(in_dim, out_dim, combine=None, add_sum=add_sum, clamp_convex_weight=clamp_convex_weight, 
                     linspace_bias=linspace_bias, bias=bias)

  def setup_linear(self,in_dim, out_dim, bias):
        return nn.Linear(in_dim, out_dim, bias=bias)

  def setup_bond_encoder(self, in_dim):
        return None
  
  def message(self, x_j):
      return self.linear(x_j)

  def aggregate(self, inputs, index):
        return inputs
  
  def update(self, aggr_out, batch=None):
        return self.adaptive_relu(aggr_out, batch)

  def forward(self, x, batch=None):
      # create edge index of self loops
      if batch is None:
          batch = torch.zeros(x.shape[0]).long().to(x.device)
      edge_index = torch.stack([torch.arange(x.shape[0]), torch.arange(x.shape[0])], dim=0).long().to(x.device)
      
      x = self.propagate(edge_index, x=x, batch=batch)
      return x