import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.conv import MessagePassing
from models.utils import norm_dict
from encoders import setup_node_encoder


class AdaptiveReLU(nn.Module):
    def __init__(self, feature_dim, add_sum=True, clamp_convex_weight=True, linspace_bias=False, bias=False):
        """
        Implementation of adaptive bias relu.
        Per multiset X, bias is chosen as t*max(<a,X>) + 1-t(<a,X>),
        where t\in [0,1] is a learned parameter.
        The function maps a multiset {{x_1,...,x_n}}->[n, Max, Min, \sum ReLU(x_i-t*Max-(1-t)*Min)],
        and then proceeds to project the 4 coordinates to a single scalar.
        This is done for each feature dimension separately.
        """
        super(AdaptiveReLU, self).__init__()

        self.feature_dim = feature_dim
        self.add_sum = add_sum
        # wether or not to clamp the convex weight parameter t to [0,1]
        self.clamp_convex_weight = clamp_convex_weight
        # differentiable function to count number of elements in each multiset
        self.bincount = lambda inds, arr: torch.scatter_reduce(input=arr, dim=0, index=inds, 
                                                               src=torch.ones_like(inds, device=arr.device), reduce="sum")

        num_coords = 5 if add_sum else 4 
        self.lin_project = nn.Linear(num_coords, 1, bias=bias)
        torch.nn.init.xavier_normal_(self.lin_project.weight)

        # Parameter t uniform from [0, 1]
        if linspace_bias:
            self.t = nn.Parameter(torch.linspace(0, 1, feature_dim))
        else:
            self.t = nn.Parameter(torch.FloatTensor(feature_dim).uniform_(0,1))

    def forward(self, x, batch_idx, max_index=None):
        """
        :param: x [num_instances, feature_dim].
        :param: batch_idx [num_instances]. instances from x that are from the same multiset share the same index.
        """
        if self.clamp_convex_weight:
            self.t.data = self.t.data.clamp(0,1)

        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0])

        num_nodes = torch.max(batch_idx)+1  if max_index is None else max_index

        # Compute min and max per batch_idx for each feature dimension
        min_values = scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='min')
        repeated_min_values = torch.zeros_like(x)
        repeated_min_values += min_values[batch_idx,:]

        max_values = scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='max')
        repeated_max_values = torch.zeros_like(x)
        repeated_max_values += max_values[batch_idx,:]

        # Compute bias as a convex combination of mins and maxs using t
        bias = self.t * repeated_max_values + (1 - self.t) * repeated_min_values

        # Add output of linear layer with bias
        translated = x - bias

        # Perform elementwise max(0, ...)
        post_relu = F.relu(translated)
        relu_sum = scatter(post_relu, batch_idx, dim=0, dim_size=num_nodes, reduce='sum')

        if self.add_sum:
          multiset_sums = scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='sum')

        # compute number of elements per multiset
        
        num_elements = self.bincount(batch_idx,
                                     torch.zeros(num_nodes, dtype=torch.long, device=x.device)).unsqueeze(-1).repeat(1, self.feature_dim)

        coords = [num_elements, min_values, max_values, relu_sum]
        if self.add_sum:
          coords.append(multiset_sums)
        output = torch.cat(coords, dim=-1)
        output = output.reshape(num_elements.shape[0], -1, self.feature_dim).permute((0,2,1))
        output = self.lin_project(output)

        return output.squeeze(-1)



class AdaptiveReLUConv(MessagePassing):
    def __init__(self, in_dim, out_dim, combine, add_sum=True, clamp_convex_weight=True, linspace_bias=False, bias=False):
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
            raise NotImplementedError('Cannot combine with LinearCombination when in_channels!=out_dim, unless using GIN')
          self.combine = LinearCombination(in_dim, 1)
        elif combine == LTSum:
          self.combine = combine(in_dim, out_dim,1)
        elif combine == Concat:
          self.combine = combine(in_dim, out_dim)
        else:
          self.combine = combine(in_dim, out_dim, out_dim)

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.adaptive_relu = AdaptiveReLU(out_dim, add_sum, clamp_convex_weight, linspace_bias, bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        with torch.no_grad():
          self.linear.weight /= torch.norm(self.linear.weight, dim=1, keepdim=True)


    def message(self, x_j):
      # project from in_channels --> out_dim
      return self.linear(x_j)


    def aggregate(self, inputs, index):
        return self.adaptive_relu(inputs, index, self.num_nodes)

    def update(self, aggr_out, orig_x):
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index):
        self.num_nodes = x.shape[0]
        x = self.propagate(edge_index, x=x, orig_x=x)
        return x
    


class AdaptiveReLUGlobalConv(AdaptiveReLUConv):
  def __init__(self, in_dim, out_dim, combine, add_sum=True, clamp_convex_weight=True, linspace_bias=False, bias=False):
    super().__init__(in_dim, out_dim, combine, add_sum, clamp_convex_weight, linspace_bias, bias)

  
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






class AdaptiveReluMPNN(nn.Module):
  def __init__(self, in_dim, embed_dim, out_dim, num_layers, combine, add_sum=True, clamp_convex_weight=True, linspace_bias=False, 
               dropout=0.0, out_mlp_layers=1, norm=None, skip_connections=False, positional_encoding=None, args=None, bias=True):
    super().__init__()
    self.in_dim = in_dim
    self.embed_dim = embed_dim
    self.out_dim = out_dim
    self.num_layers = num_layers
    self.combine = combine
    self.dropout = dropout
    self.skip_connections = skip_connections
    
    self.node_encoder = setup_node_encoder(args)
    if positional_encoding not in [None, 'none']:
        in_dim = args.embed_dim

    if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
        raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

    self.convs = nn.ModuleList()
    if num_layers > 0:
      if (combine == LinearCombination) and (in_dim!=embed_dim):
        self.convs.append(AdaptiveReLUConv(in_dim, embed_dim, LTSum, add_sum, clamp_convex_weight, linspace_bias, bias))
      else:
        self.convs.append(AdaptiveReLUConv(in_dim, embed_dim, combine, add_sum, clamp_convex_weight, linspace_bias, bias))

      if combine == Concat:
          in_dim += embed_dim
          embed_dim = in_dim

    for _ in range(num_layers-1):
      self.convs.append(AdaptiveReLUConv(embed_dim, embed_dim, combine, add_sum, clamp_convex_weight, linspace_bias, bias))
      if combine == Concat:
        in_dim += embed_dim
        embed_dim = in_dim
    
    self.norms = nn.ModuleList()
    for i in range(num_layers):
      self.norms.append(norm_dict[norm](embed_dim))

    if num_layers == 0:
       if combine == LinearCombination:
         combine = LTSum
       self.global_conv = AdaptiveReLUGlobalConv(in_dim, embed_dim, combine, add_sum, clamp_convex_weight, linspace_bias, bias)
    else:
      self.global_conv = AdaptiveReLUGlobalConv(embed_dim, embed_dim, combine, add_sum, clamp_convex_weight, linspace_bias, bias)

    out_mlp =[]
    for i in range(out_mlp_layers-1):
      out_mlp.append(nn.Linear(embed_dim, embed_dim))
      out_mlp.append(nn.ReLU())
    out_mlp.append(nn.Linear(embed_dim, out_dim))
    self.out_MLP = nn.Sequential(*out_mlp)


  def forward(self, batch, return_node_embeddings=False):
    self.node_encoder(batch)
    x, edge_index = batch.x, batch.edge_index
    for i, conv_layer in enumerate(self.convs):
        h = F.dropout(self.norms[i](conv_layer(x, edge_index)), p=self.dropout, training=self.training)
        if self.skip_connections and i > 0:
          x = h + x
        else:
          x = h
    
    embedding = self.global_conv(x, batch.batch)

    global_out = self.out_MLP(embedding)
    if return_node_embeddings:
      return global_out, embedding, x

    return global_out, embedding
  




  
# test
if __name__ == '__main__':
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    import argparse
    parser = argparse.ArgumentParser()
    embed_dim = 212
    encoder_kwargs = {
        'embed_dim':embed_dim,
        'rw_steps':20,
        'lape_k':10,
        'positional_encoding':None,
        'pe_embed_dim':28,
        'in_dim':9
    }

    args = parser.parse_args([])
    for key, value in encoder_kwargs.items():
        setattr(args, key, value)
    in_dim = 9
    
    out_dim = 10
    num_layers = 1
    combine = LinearCombination
    add_sum=True
    clamp_convex_weight=True
    linspace_bias=True, 
    dropout = 0
    out_mlp_layers = 3
    norm = None
    positional_encoding = None
    skip_connections = False
    model = AdaptiveReluMPNN(in_dim, embed_dim, out_dim, num_layers, combine, add_sum, 
                             clamp_convex_weight, linspace_bias, dropout, out_mlp_layers, norm, 
                             skip_connections, positional_encoding, args)
    
    print(f'Number of parameters: {count_params(model)}')

    x = torch.randn(10, in_dim)
    src = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long()
    dst = torch.Tensor([1,2,3,4,0,6,7,8,9,5]).long()
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    batch = torch.zeros(10).long()
    from torch_geometric.data import Data
    batch = Data(x=x, edge_index=edge_index, batch=batch)

    out = model(batch, return_node_embeddings=True)
    print(out[0].shape, out[1].shape, out[2].shape)

