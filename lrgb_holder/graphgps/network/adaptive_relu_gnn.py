import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from graphgps.layer.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.norm import LayerNorm, BatchNorm
combine_dict = {'LinearCombination': LinearCombination, 'LTSum': LTSum, 'Concat': Concat, 'ConcatProject': ConcatProject}
norm_dict = {'layer': LayerNorm, 'batch': BatchNorm, 'none': torch.nn.Identity}
from graphgps.layer.adaptive_relu_layer import AdaptiveReLUConv
import torch.nn as nn
import torch.nn.functional as F



@register_network('adaptive_relu_gnn')
class AdaptiveReluMPNN(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    embed_dim = cfg.gnn.dim_inner
    num_layers = cfg.gnn.layers_mp
    combine = combine_dict[cfg.gnn.combine]
    residual = cfg.gnn.residual
    self.dropout = cfg.gnn.dropout
    norm = cfg.gnn.norm
    self.skip_connections = residual
    self.dim_in = dim_in
    self.embed_dim = embed_dim
    self.dim_out = dim_out
    self.num_layers = num_layers
    self.combine = combine
    
    if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
        raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

    self.node_encoder = FeatureEncoder(dim_in)
    dim_in = self.node_encoder.dim_in

    if cfg.gnn.layers_pre_mp > 0:
      self.pre_mp = GNNPreMP(
        dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
      dim_in = cfg.gnn.dim_inner

    assert cfg.gnn.dim_inner == dim_in, \
      "The inner and hidden dims must match."

    first_combine = LTSum if (combine == LinearCombination and dim_in!=embed_dim)  else combine
    conv1 = AdaptiveReLUConv(dim_in, embed_dim, combine=first_combine)
    layers = [conv1]
    for _ in range(cfg.gnn.layers_mp-1):
        layers.append(AdaptiveReLUConv(dim_in, dim_in, combine=combine))
    self.convs = torch.nn.Sequential(*layers)
    self.norms = nn.ModuleList()
    for i in range(num_layers):
      self.norms.append(norm_dict[norm](embed_dim))
    
    GNNHead = register.head_dict[cfg.gnn.head]
    self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
    


  def forward(self, batch, return_node_embeddings=False):
    self.node_encoder(batch)
    x, edge_index = batch.x, batch.edge_index
    
    for i, conv_layer in enumerate(self.convs):
        h = F.dropout(self.norms[i](conv_layer(x, edge_index)), p=self.dropout, training=self.training)
        if self.skip_connections and i > 0:
          x = h + x
        else:
          x = h
     
    batch.x = x
    out = self.post_mp(batch)

    return out
