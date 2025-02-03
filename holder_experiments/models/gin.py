import torch.nn as nn
import torch
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch.nn import Linear
from torch_geometric.nn.models import GIN


class GINWrapper(nn.Module):
  '''
  GIN with specified number of layers.
  '''
  def __init__(self, num_layers, in_dim, embed_dim, out_dim, activation=nn.ReLU(), dropout=0, out_mlp_layers=1,
               norm=None, jk=None):
    super().__init__()
    self.num_layers = num_layers
    self.activation = activation
    self.dropout = nn.Dropout(dropout)
    self.convs = GIN(in_channels=in_dim, hidden_channels=embed_dim, out_channels=embed_dim, num_layers=num_layers, dropout=dropout,
                     norm=norm, jk=jk)

    out_mlp = []
    for _ in range(out_mlp_layers-1):
      out_mlp.append(nn.Linear(embed_dim, embed_dim))
      out_mlp.append(activation)
      out_mlp.append(self.dropout)
    out_mlp.append(nn.Linear(embed_dim, out_dim))
    self.out_mlp = nn.Sequential(*out_mlp)
    

  def forward(self, batch, return_node_embeddings=False):
    x, edge_index = batch.x, batch.edge_index
    x = self.convs(x, edge_index)
    embedding = global_add_pool(x, batch.batch)
    output = self.out_mlp(embedding)

    if return_node_embeddings:
      return output, embedding, x

    return output, embedding
