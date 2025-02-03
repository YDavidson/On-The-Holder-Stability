
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import torch
# import mutag dataset
from torch_geometric.datasets import TUDataset
import os
import sys 
# add parent directory to path when running vscode
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_relu_mpnn import AdaptiveReluMPNN
from models.sort_mpnn import SortMPNN
from models.mlp_moments import MLPMomentMPNN
from models.gin import GIN
from models.combine import LTSum, LinearCombination, Concat, ConcatProject
from utils import get_max_degree_and_nodes

def test_invariance(model, graph_1, graph_2):
    # graph_1 and graph_2 are Data objects
    model.eval()
    with torch.no_grad():
        _, embedding_1 = model(graph_1.x, graph_1.edge_index)
        _, embedding_2 = model(graph_2.x, graph_2.edge_index)
    assert torch.allclose(embedding_1, embedding_2) == True, 'Graph Embeddings are not invariant'

def test_equivariance(model, graph_1, graph_2, perm_mat):
    # graph_1 and graph_2 are Data objects
    model.eval()
    with torch.no_grad():
        _, _, embeddings_1 = model(graph_1.x, graph_1.edge_index, return_node_embeddings=True)
        _, _, embeddings_2 = model(graph_2.x, graph_2.edge_index, return_node_embeddings=True)
    assert torch.allclose(perm_mat@embeddings_1, embeddings_2) == True, 'Node Embeddings are not equivariant'



# load mutag dataset
dataset = TUDataset(root='DATA', name='MUTAG')

graph_1 = dataset[0]

# get adjacency matrix of graph_1
adj_1 = to_dense_adj(graph_1.edge_index)[0]
# create permutation matrix
perm = torch.randperm(adj_1.shape[0])
perm_mat = torch.zeros_like(adj_1)
perm_mat[perm, torch.arange(perm.shape[0])] = 1

# permute adjacency matrix
adj_2 = perm_mat @ adj_1 @ perm_mat.T
# create edge_index of permuted adjacency matrix
edge_index_2, _ = dense_to_sparse(adj_2)

# permute node features
x_2 = perm_mat @ graph_1.x

# create graph_2
graph_2 = Data(x=x_2, edge_index=edge_index_2)

# general model params
in_dim = graph_1.x.shape[1]
embed_dim = 8
out_dim = dataset.num_classes
num_layers = 3

# create sort model
bias = False
max_neighbors, max_nodes = get_max_degree_and_nodes(dataset)
sort_model = SortMPNN(in_dim, embed_dim, out_dim, num_layers, bias, max_neighbors, max_nodes, combine=LinearCombination)

# create relu mlp model
relu_model = MLPMomentMPNN(in_dim, embed_dim, out_dim, num_layers, combine=LinearCombination)


# create adaptive relu model
adaptive_relu_model = AdaptiveReluMPNN(in_dim, embed_dim, out_dim, num_layers, combine=LinearCombination)


for model in [sort_model, relu_model, adaptive_relu_model]:
    # test invariance
    test_invariance(model, graph_1, graph_2)

    # test equivariance
    test_equivariance(model, graph_1, graph_2, perm_mat)








    