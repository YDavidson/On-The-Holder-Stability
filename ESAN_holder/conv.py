"""
Code taken from ogb examples and adapted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_batch
from combine import LTSum, LinearCombination, Concat, ConcatProject
import torch_scatter


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
        self.bincount = lambda inds, arr: torch.scatter_add(input=arr, dim=0, index=inds, 
                                                               src=torch.ones_like(inds, device=arr.device))

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
        min_values = torch_scatter.scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='min')
        repeated_min_values = torch.zeros_like(x)
        repeated_min_values += min_values[batch_idx,:]

        max_values = torch_scatter.scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='max')
        repeated_max_values = torch.zeros_like(x)
        repeated_max_values += max_values[batch_idx,:]

        # Compute bias as a convex combination of mins and maxs using t
        bias = self.t * repeated_max_values + (1 - self.t) * repeated_min_values

        # Add output of linear layer with bias
        translated = x - bias

        # Perform elementwise max(0, ...)
        post_relu = F.relu(translated)
        relu_sum = torch_scatter.scatter(post_relu, batch_idx, dim=0, dim_size=num_nodes, reduce='sum')

        if self.add_sum:
          multiset_sums = torch_scatter.scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='sum')

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



class ZINCAdaptiveReLUConv(MessagePassing):
    def __init__(self, in_dim, out_dim, combine, add_sum=True, clamp_convex_weight=True, 
                 linspace_bias=False, bias=False, relu_edge=False):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param combine: combine scheme to use from [LinearCombination|LTSum|Concat|ConcatProject]
        """
        super().__init__()
        self.bond_encoder = self.setup_bond_encoder(in_dim)

        try:
            combine = combine_dict[combine]
        except:
           raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu_edge = relu_edge
        
        if combine == LinearCombination:
          if in_dim != out_dim:
            self.combine = LTSum(in_dim, out_dim,1)
          else:
            self.combine = combine(in_dim, 1)
        elif combine == LTSum:
          self.combine = combine(in_dim, out_dim,1)
        elif combine == Concat:
          self.combine = combine(in_dim, out_dim)
        elif combine == ConcatProject:
          self.combine = combine(in_dim, out_dim, out_dim)
        elif combine is None:
          self.combine = torch.nn.Identity()
        else:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

        self.linear = self.setup_linear(in_dim, out_dim, bias)
        self.adaptive_relu = AdaptiveReLU(out_dim, add_sum, clamp_convex_weight, linspace_bias, bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        

    def setup_linear(self,in_dim, out_dim, bias):
        return nn.Linear(2*in_dim, out_dim, bias=bias)


    def setup_bond_encoder(self, in_dim):
        return torch.nn.Embedding(4, in_dim)

    def message(self, x_j, edge_attr):
        if self.relu_edge:
            return F.relu(self.linear(torch.cat([x_j, edge_attr], dim=-1)))
        else:
            return self.linear(torch.cat([x_j, edge_attr], dim=-1))
      
    def aggregate(self, inputs, index):
        return self.adaptive_relu(inputs, index, self.num_nodes)

    def update(self, aggr_out, orig_x):
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        self.num_nodes = x.shape[0]
        x = self.propagate(edge_index, x=x, orig_x=x, edge_attr=edge_embedding)
        return x



class ZINCSortConv(MessagePassing):
    def __init__(self, in_dim, out_dim, max_degree, bias=True, combine='LinearCombination', 
                 collapse_method='vector', blank_method='learnable', relu_edge=False):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param max_degree: max number of neighbors across all dataset
        """
        super().__init__()
        self.relu_edge = relu_edge
        self.bond_encoder = torch.nn.Embedding(4, in_dim)
        self.bond_encoder = self.setup_bond_encoder(in_dim)
        self.lin_project = self.setup_projection(in_dim, out_dim, bias)

        # Setting up for chosen collapse method
        self.collapse_method = collapse_method
        if collapse_method == 'vector':
          self.lin_collapse = torch.nn.Linear(max_degree, 1, bias=bias)
        elif collapse_method == 'matrix':
          self.lin_collapse = torch.nn.Parameter(torch.zeros((max_degree,out_dim)))
        else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

        # Setting up chosen update method
        try:
            combine = combine_dict[combine]
        except:
           raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')
        if combine == LinearCombination:
          if in_dim != out_dim:
            # raise NotImplementedError('Cannot combine with LinearCombination when in_dim!=out_dim')
            self.combine = LTSum(in_dim, out_dim,1)
          else:
            self.combine = combine(in_dim, 1)
        elif combine == LTSum:
          self.combine = combine(in_dim, out_dim,1)
        elif combine == Concat:
          self.combine = combine(in_dim, out_dim)
        elif combine == ConcatProject:
          self.combine = combine(in_dim, out_dim, out_dim)
        elif combine is None:
          self.combine = torch.nn.Identity()
        else:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

        
        # setting up blank vector
        if blank_method == 'learnable':
           blank_grad = True
        elif blank_method == 'zero':
           blank_grad = False
        else:
           raise NotImplementedError('blank_method must be one of [learnable|zero]')
        self.blank_vec = torch.nn.parameter.Parameter(torch.zeros(out_dim, requires_grad=blank_grad))
           

        self.max_degree = max_degree
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin_project.weight)
        # Normalize weights so they are on the unit sphere
        self.lin_project.weight = nn.Parameter(self.lin_project.weight / torch.norm(self.lin_project.weight, dim=1, keepdim=True))
        if self.collapse_method == 'vector':
          torch.nn.init.xavier_normal_(self.lin_collapse.weight)
        elif self.collapse_method == 'matrix':
          torch.nn.init.normal_(self.lin_collapse)


    def setup_projection(self, in_dim, out_dim, bias):
       return torch.nn.Linear(2*in_dim, out_dim, bias=bias)
        
    def setup_bond_encoder(self, in_dim):
       return torch.nn.Embedding(4, in_dim)


    def get_num_indices(self, index):
      with torch.no_grad():
        num_indices = int(torch.max(index).item()) +1
        return num_indices

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1) # [Num_nodes("batch"), num_neighbors("max_degree"), out_dim] --> [Num_nodes("batch"), out_dim]
      elif self.collapse_method == 'matrix':
        out = sorted * self.lin_collapse
        out = torch.sum(out, dim=1).squeeze(1)
      else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

      return out
    
    def replace_indices_(self, index):
      return index
    
    def convert_to_dense(self, inputs, index, indices, num_indices):
        dense, mask = to_dense_batch(inputs[indices], index[indices],max_num_nodes=self.max_degree)
        if dense.shape[0] < num_indices:
           extra_rows = num_indices - dense.shape[0]
           dense = torch.cat([dense, torch.zeros((extra_rows, dense.shape[1], dense.shape[2]), device=dense.device)], dim=0)
           mask = torch.cat([mask, torch.zeros((extra_rows, mask.shape[1]), device=mask.device).to(torch.bool)], dim=0)
        return dense, mask
       
    def message(self, x_j, edge_attr):
        if self.relu_edge:
            return F.relu(self.lin_project(torch.cat([x_j, edge_attr], dim=-1)))
        else:
            return self.lin_project(torch.cat([x_j, edge_attr], dim=-1))


    def aggregate(self, inputs, index):
      # in index, we replace each index with its position in the sorted list
      index = self.replace_indices_(index)

      # create dense neighborhoods augmented with blank vectors
      num_indices = self.num_indices
      _, indices = torch.sort(index)
      result, mask = self.convert_to_dense(inputs, index, indices, num_indices)
      
      result += (~mask.unsqueeze(-1)).repeat(1,1,self.blank_vec.shape[0])*self.blank_vec # filling with blank tree vector
      
      # sort each column independently
      sorted, _ = torch.sort(result, dim=-2)

      # collapse
      return self.return_collapsed_output(sorted)


    def update(self, aggr_out, orig_x):
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        orig_x = x
        self.num_indices = x.shape[0]
        x = self.propagate(edge_index, x=x, orig_x=orig_x, edge_attr=edge_embedding)
        return x



### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


combine_dict = {'LinearCombination': LinearCombination,
                'LTSum': LTSum,
                'ConcatProject': ConcatProject,
                'Concat': Concat,
                None: None}




class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.layer = PyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 num_random_features=0, feature_encoder=lambda x: x, conv_kwargs=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        ### add residual connection or not
        self.residual = residual
        self.gnn_type = gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = feature_encoder
        self.num_random_features = num_random_features

        if num_random_features > 0:
            assert gnn_type == 'graphconv'

            self.initial_layers = torch.nn.ModuleList(
                [GraphConv(in_dim, emb_dim // 2), GraphConv(emb_dim // 2, emb_dim - num_random_features)]
            )
            # now the next layers will have dimension emb_dim
            in_dim = emb_dim

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'originalgin':
                self.convs.append(OriginalGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'zincgin':
                self.convs.append(ZINCGINConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'graphconv':
                self.convs.append(GraphConv(emb_dim if layer != 0 else in_dim, emb_dim))
            elif gnn_type == 'sort':
                self.convs.append(ZINCSortConv(emb_dim if layer != 0 else in_dim, emb_dim, **conv_kwargs))
            elif gnn_type == 'adaptive_relu':
                self.convs.append(ZINCAdaptiveReLUConv(emb_dim if layer != 0 else in_dim, emb_dim, **conv_kwargs))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.num_random_features > 0:
            for layer in self.initial_layers:
                x = F.elu(layer(x, edge_index, edge_attr))

            # Implementation of RNI
            random_dims = torch.empty(x.shape[0], self.num_random_features).to(x.device)
            torch.nn.init.normal_(random_dims)
            x = torch.cat([x, random_dims], dim=1)

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)

            if self.gnn_type not in ['gin', 'gcn'] or layer != self.num_layer - 1:
                h = F.relu(h)  # remove last relu for ogb

            if self.drop_ratio > 0.:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            start_idx = 0 if self.in_dim == self.emb_dim else 1
            for layer in range(start_idx, self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)

        return node_representation
