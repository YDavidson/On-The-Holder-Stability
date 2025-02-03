import torch
from torch_geometric.utils import to_dense_batch
import os
import sys
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.graphgym.config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graphgps.layer.adaptive_relu_layer import AdaptiveReLUConv





@register_pooling('adaptive_relu_global')
class AdaptiveReLUGlobalConv(AdaptiveReLUConv):
  def __init__(self, in_dim, out_dim, combine=None):
    super().__init__(in_dim, out_dim, combine)

  
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
    
