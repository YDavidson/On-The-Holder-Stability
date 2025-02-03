import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, index_sort, to_dense_batch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.conv import MessagePassing
from models.utils import norm_dict
from encoders import setup_node_encoder




class SortConv(MessagePassing):
    def __init__(self, in_dim, out_dim, max_nodes, bias=False, combine=LTSum, collapse_method='vector', orig_dim=None):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param max_nodes: max number of neighbors across all dataset
        """
        super().__init__()

        if orig_dim is None:
          orig_dim = in_dim
        self.orig_dim = orig_dim
        self.out_dim = out_dim

        self.lin_project = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Setting up for chosen collapse method
        self.collapse_method = collapse_method

        if collapse_method == 'vector':
          self.lin_collapse = torch.nn.Linear(max_nodes, 1, bias=bias)
        elif collapse_method == 'matrix':
          self.lin_collapse = torch.nn.Parameter(torch.zeros((max_nodes,out_dim)))
        else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

        # Setting up chosen combine method
        if combine == LinearCombination:
          if in_dim != out_dim:
            raise NotImplementedError('Cannot combine with LinearCombination when in_dim!=out_dim')
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

        self.max_nodes = max_nodes
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin_project.weight)
        self.lin_project.weight = nn.Parameter(self.lin_project.weight / torch.norm(self.lin_project.weight, dim=1, keepdim=True))
        if self.collapse_method == 'vector':
          torch.nn.init.xavier_normal_(self.lin_collapse.weight)
        elif self.collapse_method == 'matrix':
          torch.nn.init.normal_(self.lin_collapse)

    def get_num_indices(self, index):
      with torch.no_grad():
        num_indices = int(torch.max(index).item()) +1
        return num_indices

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1) 
      elif self.collapse_method == 'matrix':
        out = sorted * self.lin_collapse
        out = torch.sum(out, dim=1).squeeze(1)
      else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

      self.blank_vec = out[-1:,:]
      return out[:-1,:]
    
    def replace_indices_(self, index):
      return index
    
    def convert_to_dense(self, inputs, index, indices, num_indices):
       return to_dense_batch(inputs[indices], index[indices],
                                    max_num_nodes=self.max_nodes, batch_size=num_indices+1)
       
    def message(self, x_j):
      return self.lin_project(x_j)
       
    def aggregate(self, inputs, index, blank_vec):
      # in index, we replace each index with its position in the sorted list
      index = self.replace_indices_(index)

      # create dense neighborhoods augmented with blank vectors
      num_indices = self.num_indices
      _, indices = index_sort(index)
      # converting to dense batch, adding blank vector to end
      result, mask = self.convert_to_dense(inputs, index, indices, num_indices)
      ## result expected to be of shape [N, max_nodes, out_dim], where N is number of indices + 1
      blank_vec = self.lin_project(blank_vec)
      result += (~mask.unsqueeze(-1)).repeat(1,1,blank_vec.shape[0])*blank_vec # filling with blank tree vector

      # sort each column independently
      sorted, _ = torch.sort(result, dim=-2)

      # collapse
      return self.return_collapsed_output(sorted)


    def update(self, aggr_out, orig_x, blank_vec, orig_blank_vec):
        if len(blank_vec.shape) < 2:
          blank_vec = blank_vec.unsqueeze(0)
        if len(orig_blank_vec.shape) < 2:
          orig_blank_vec = orig_blank_vec.unsqueeze(0)

        self.blank_vec = self.combine(orig_blank_vec,self.blank_vec)
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index, blank_vec, orig_x=None, orig_blank_vec=None):
        if orig_x is None:
          orig_x = x
        if orig_blank_vec is None:
          orig_blank_vec = blank_vec
        self.num_indices = x.shape[0]

        x = self.propagate(edge_index, x=x, blank_vec=blank_vec, orig_x=orig_x, 
                           orig_blank_vec=orig_blank_vec)
        return x, self.blank_vec



class SortGlobalConv(SortConv):
    def __init__(self, in_dim, out_dim, max_nodes, bias=False, collapse_method='vector'):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param max_nodes: max number of vertices in graph across all dataset
        """
        super().__init__(in_dim, out_dim, max_nodes, bias, collapse_method=collapse_method, combine=None)

    def convert_to_dense(self, inputs, index, indices, num_indices):
       return to_dense_batch(inputs[indices], index[indices],
                                    max_num_nodes=self.max_nodes, batch_size=num_indices)

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1)
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

    def update(self, aggr_out, index, batch):
        # return the aggregated out
        return aggr_out
    

    def forward(self, x, blank_vec, batch=None):
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

        x = self.propagate(edge_index, x=x, blank_vec=blank_vec, batch=batch)
        return x
    

def setup_learnable_blank_vector(in_dim, embed_dim, positional_encoding, num_layers):
    if positional_encoding not in [None, 'none']:
        in_dim = embed_dim
    blank_vectors = nn.ParameterList([nn.Parameter(torch.zeros(in_dim))])
    for _ in range(num_layers):
        blank_vectors.append(nn.Parameter(torch.zeros(embed_dim)))

    return blank_vectors

class SortMPNN(torch.nn.Module):
    '''
    MPNN that uses sort as non-linearity.
    The MPNN optionally maintains an additional blank tree vector which augments node neighborhoods.
    '''
    def __init__(self, in_dim, embed_dim, out_dim, num_layers, bias, max_neighbors, max_nodes, 
                 combine=LTSum, collapse_method='matrix', update_w_orig=False, dropout=0, out_mlp_layers=1,
                 norm=None, skip_connections=False, positional_encoding=None, args=None, out_mlp_dropout=False, 
                 blank_vector_method='iterative_update'):
        super().__init__()
        if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')
        
        assert blank_vector_method in ['iterative_update', 'zero', 'learnable'], \
                                      'blank_vector_method must be one of [iterative_update, zero, learnable]'
        self.blank_vector_method = blank_vector_method
        if self.blank_vector_method == 'learnable':
          self.blank_vectors = setup_learnable_blank_vector(in_dim, embed_dim, positional_encoding, num_layers)

        self.update_w_orig = update_w_orig
        self.dropout = dropout
        self.skip_connections = skip_connections
        
        self.node_encoder = setup_node_encoder(args)
        if positional_encoding not in [None, 'none']:
            in_dim = args.embed_dim

        orig_dim = in_dim if update_w_orig else None
        if num_layers > 0:
          first_combine = LTSum if (combine == LinearCombination and in_dim!=embed_dim)  else combine
          conv1 = SortConv(in_dim, embed_dim, max_neighbors, bias, combine=first_combine, collapse_method=collapse_method, orig_dim=orig_dim)
          if combine == Concat:
            embed_dim = embed_dim + in_dim
          self.convs = [conv1]
        else:
           self.convs = []
        for _ in range(num_layers-1):
          self.convs.append(
              SortConv(embed_dim, embed_dim, max_neighbors, bias, combine=combine, collapse_method=collapse_method, orig_dim=orig_dim)
          )
          if combine == Concat:
            embed_dim += embed_dim

        self.convs = nn.ModuleList(self.convs)

        self.norms = nn.ModuleList()
        for i in range(num_layers):
          self.norms.append(norm_dict[norm](embed_dim))
        
        if num_layers == 0:
          self.global_conv = SortGlobalConv(in_dim, embed_dim, max_nodes, bias, collapse_method=collapse_method)
        else:
          self.global_conv = SortGlobalConv(embed_dim, embed_dim, max_nodes, bias, collapse_method=collapse_method)

        out_mlp =[]
        for i in range(out_mlp_layers-1):
          if out_mlp_dropout:
            out_mlp.append(nn.Dropout(dropout))
          out_mlp.append(nn.Linear(embed_dim, embed_dim))
          out_mlp.append(nn.ReLU())
        if out_mlp_dropout:
          out_mlp.append(nn.Dropout(dropout))
        out_mlp.append(nn.Linear(embed_dim, out_dim))
        self.out_MLP = nn.Sequential(*out_mlp)


    def forward(self, batch, return_node_embeddings=False):
        self.node_encoder(batch)
        x ,edge_index = batch.x, batch.edge_index
        orig_x = x if self.update_w_orig else None
        if self.blank_vector_method == 'learnable':
          blank_vec = self.blank_vectors[0].to(x.device)
        else:
          blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
        orig_blank_vec = blank_vec if self.update_w_orig else None

        for i, conv in enumerate(self.convs):
            if self.blank_vector_method == 'learnable':
              blank_vec = self.blank_vectors[i].to(x.device)
            elif self.blank_vector_method == 'zero':
              blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
            
            if self.skip_connections:
               res_x , res_blank_vec = x, blank_vec
            x, blank_vec = conv(x, edge_index, blank_vec, orig_x=orig_x, orig_blank_vec=orig_blank_vec)
            
                        
            # apply dropout to both x and blank_vec
            x = F.dropout(self.norms[i](x), p=self.dropout, training=self.training)
            blank_vec = F.dropout(blank_vec, p=self.dropout, training=self.training)

            if self.skip_connections and i > 0:
              x = x + res_x
              blank_vec = blank_vec + res_blank_vec
            
        if self.blank_vector_method == 'learnable':
          blank_vec = self.blank_vectors[-1].to(x.device)
        elif self.blank_vector_method == 'zero':
          blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)

        embedding = self.global_conv(x, blank_vec, batch.batch)
        out = self.out_MLP(embedding)

        if return_node_embeddings:
          return out, embedding, x

        return out, embedding
    



