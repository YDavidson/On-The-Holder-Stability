import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from models.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.conv import MessagePassing
from models.utils import norm_dict


class MLPMomentConv(MessagePassing):
    def __init__(self, in_dim, out_dim, combine, activation=nn.ReLU, bias_range=None, linspace_bias=False):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param combine: combine scheme to use from [LinearCombination|LTSum|Concat|ConcatProject]
        """
        super().__init__()

        if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

        self.in_dim = in_dim
        self.out_dim = out_dim

        if combine == LinearCombination:
          if in_dim != out_dim:
            raise NotImplementedError('Cannot combine with LinearCombination when in_channels!=out_dim')
          self.combine = LinearCombination(in_dim, 1)
        elif combine == LTSum:
          self.combine = combine(in_dim, out_dim,1)
        elif combine == Concat:
          self.combine = combine(in_dim, out_dim)
        else:
          self.combine = combine(in_dim, out_dim, out_dim)


        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation(),
        )

        bias_range = (-2.0, 2.0) if bias_range is None else (-bias_range, bias_range)
        self.reset_parameters(bias_range, linspace_bias)

    def reset_parameters(self, bias_range, linspace_bias):
        torch.nn.init.xavier_normal_(self.MLP[0].weight)
        # Normalize weights so they are on the unit sphere
        with torch.no_grad():
          self.MLP[0].weight /= (torch.norm(self.MLP[0].weight, dim=1, keepdim=True))
        # # set bias
        if linspace_bias:
          # euqally spaced bias values
          self.MLP[0].bias = torch.nn.Parameter(torch.linspace(bias_range[0], bias_range[1], self.out_dim))
        else:
          torch.nn.init.uniform_(self.MLP[0].bias, bias_range[0], bias_range[1])



    def message(self, x_j):
      # project from in_channels --> out_dim
      return self.MLP(x_j)


    def update(self, aggr_out, orig_x):
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x, orig_x=x)
        return x
    


class MLPMomentGlobalConv(MLPMomentConv):
  def __init__(self, in_dim, out_dim, combine, activation=nn.ReLU, bias_range=None, linspace_bias=False):
    super().__init__(in_dim, out_dim, combine, activation, bias_range, linspace_bias)

  def update(self, aggr_out, batch=None):
        # create edge index of self loops
        if batch is None:
            batch = torch.zeros(aggr_out.shape[0]).long().to(aggr_out.device)

        # multiply central node by eta and add to aggregated neighbors
        return scatter(aggr_out, batch, dim=0, reduce='add')

  def forward(self, x, batch=None):
      # create edge index of self loops
      if batch is None:
          batch = torch.zeros(x.shape[0]).long().to(x.device)
      edge_index = torch.stack([torch.arange(x.shape[0]), torch.arange(x.shape[0])], dim=0).long().to(x.device)

      x = self.propagate(edge_index, x=x, batch=batch)
      return x




class MLPMomentMPNN(nn.Module):
  def __init__(self, in_dim, embed_dim, out_dim, num_layers, combine, activation=nn.ReLU,
               bias_ranges=None, linspace_bias=False, dropout=0, out_mlp_layers=1, norm=None,
               skip_connections=False):
    super().__init__()
    self.in_dim = in_dim
    self.embed_dim = embed_dim
    self.out_dim = out_dim
    self.num_layers = num_layers
    self.combine = combine
    self.dropout = dropout
    self.skip_connections = skip_connections
    
    if bias_ranges is None:
      bias_ranges = [None for i in range(num_layers + 1)]
    elif len(bias_ranges) != num_layers + 1:
      raise NotImplementedError("When bias ranges are given, must be of length num_layers + 1")


    if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

    self.norms = nn.ModuleList()
    for i in range(num_layers):
      self.norms.append(norm_dict[norm](embed_dim))

    self.convs = nn.ModuleList()
    if self.num_layers > 0:
      if (combine == LinearCombination) and (in_dim!=embed_dim):
        self.convs.append(MLPMomentConv(in_dim, embed_dim, LTSum, activation, bias_ranges[0], linspace_bias))
      else:
        self.convs.append(MLPMomentConv(in_dim, embed_dim, combine, activation, bias_ranges[0], linspace_bias))

      if combine == Concat:
          in_dim += embed_dim
          embed_dim = in_dim

    for  i in range(1, num_layers):
      self.convs.append(MLPMomentConv(embed_dim, embed_dim, combine, activation, 
                                      bias_ranges[i], linspace_bias))
      if combine == Concat:
        in_dim += embed_dim
        embed_dim = in_dim
    
    if num_layers == 0:
       if combine == LinearCombination:
          combine = LTSum
       self.global_conv = MLPMomentGlobalConv(in_dim, embed_dim, combine, activation, bias_ranges[-1], linspace_bias)
    else:
      self.global_conv = MLPMomentGlobalConv(embed_dim, embed_dim, combine, activation, bias_ranges[-1], linspace_bias)

    out_mlp =[]
    for i in range(out_mlp_layers-1):
      out_mlp.append(nn.Linear(embed_dim, embed_dim))
      out_mlp.append(activation())
    out_mlp.append(nn.Linear(embed_dim, out_dim))
    self.out_MLP = nn.Sequential(*out_mlp)

  def forward(self, batch, return_node_embeddings=False):
    x, edge_index = batch.x, batch.edge_index
    for i, conv_layer in enumerate(self.convs):
        if self.skip_connections and i > 0:
          x = F.dropout(self.norms[i](conv_layer(x, edge_index)), p=self.dropout, training=self.training) + x
        else:
          x = F.dropout(self.norms[i](conv_layer(x, edge_index)), p=self.dropout, training=self.training) 
        
    
    embedding = self.global_conv(x, batch.batch)

    global_out = self.out_MLP(embedding)
    if return_node_embeddings:
      return global_out, embedding, x

    return global_out, embedding
  

