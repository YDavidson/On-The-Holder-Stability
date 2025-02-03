import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_layer


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, index_sort, to_dense_batch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graphgps.layer.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.conv import MessagePassing
combine_dict = {'LinearCombination': LinearCombination, 'LTSum': LTSum, 'Concat': Concat, 'ConcatProject': ConcatProject}



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


@register_layer('adaptive_relu_conv')
class AdaptiveReLUConv(MessagePassing):
    def __init__(self, in_dim, out_dim, combine=None):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param combine: combine scheme to use from [LinearCombination|LTSum|Concat|ConcatProject]
        """
        super().__init__()
        combine = combine_dict[cfg.gnn.combine] if combine is None else combine
        add_sum = cfg.gnn.add_sum
        clamp_convex_weight = cfg.gnn.clamp_convex_weight
        linspace_bias = cfg.gnn.linspace_bias
        bias = cfg.gnn.bias
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')        

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
    
