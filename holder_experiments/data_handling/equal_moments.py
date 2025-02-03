import numpy as np
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import files_exist
import numpy as np
import os
from typing import Sequence, Type

class EqualMomentsDatasets(InMemoryDataset):
    def __init__(self, root, k, num_pairs, min_epsilon, max_epsilon, seed=42, force_reload=True, add_outliers=False):
        self.k = k
        self.num_pairs = num_pairs
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.seed = seed
        self.add_outliers = add_outliers

        if force_reload and files_exist([os.path.join(root, 'processed', f'processed_data_k_{self.k}.pt')]):
            # if data exists, delete it
            os.remove(os.path.join(root, 'processed', f'processed_data_k_{self.k}.pt'))

        super(EqualMomentsDatasets, self).__init__(root, None, None)
        self.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'processed_data_k_{self.k}.pt']

    def download(self):
        pass

    def process(self):
        self.data = []
        x, y = self.arbitrary_moments(self.k)
        if self.add_outliers:
            x = np.concatenate([x, np.array([2.0,-2.0])])
            y = np.concatenate([y, np.array([2.0,-2.0])])
        num_elements = len(x)
        base_multiset = np.zeros(num_elements)
        # edges = torch.tensor([[i for i in range(num_elements)], [num_elements for _ in range(num_elements)]])
        # edges = torch.cat([edges, edges.flip(dims=[0])], dim=-1)
        edges = torch.tensor([[], []], dtype=torch.long)
        for eps in np.linspace(self.min_epsilon, self.max_epsilon, self.num_pairs):
            shifted_x = base_multiset + eps*x
            shifted_y = base_multiset + eps*y
            if self.add_outliers:
                shifted_x[-2:] = np.array([2.0, -2.0])
                shifted_y[-2:] = np.array([2.0, -2.0])

            # x_features = torch.cat([torch.tensor(shifted_x, dtype=torch.float32), 
            #                         torch.tensor([0], dtype=torch.float32)], dim=0).unsqueeze(-1)
            # y_features = torch.cat([torch.tensor(shifted_y, dtype=torch.float32), 
            #                         torch.tensor([0], dtype=torch.float32)], dim=0).unsqueeze(-1)
            x_features = torch.tensor(shifted_x, dtype=torch.float32).unsqueeze(-1)
            y_features = torch.tensor(shifted_y, dtype=torch.float32).unsqueeze(-1)
            
            graph_x = Data(x=x_features, edge_index=edges, y=torch.tensor([0], dtype=torch.long))
            graph_y = Data(x=y_features, edge_index=edges, y=torch.tensor([1], dtype=torch.long))
            self.data.append(graph_x)
            self.data.append(graph_y)

        self.save(self.data, self.processed_paths[0])

    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        torch.save((data.to_dict(), slices, data.__class__), path)

    def load(self, path: str, data_cls: Type[BaseData] = Data):
        r"""Loads the dataset from the file path :obj:`path`."""
        data, self.slices, _ = torch.load(path)
        if isinstance(data, dict):  # Backward compatibility.
            data = data_cls.from_dict(data)
        self.data = data    

    @ staticmethod
    def arbitrary_moments(k):
        """
        Generate two distinct sorted vectors x,y of length 2^{k+1} whose first 2^k moments are the same
        """
        i=0
        #two vectors of length 2 whose first moment are the same
        x=np.array([0,2])
        y=np.array([1,1])
        while i<k:
            #translate x and y by m so that they are both positive, still have same 2^i moments
            m=np.min([x,y])
            x=x-m
            y=y-m
            #take x=[rx,-rx] and y=[ry,-ry]. They will have 2^{i+1} same moments
            rx=np.sqrt(x)
            ry=np.sqrt(y)
            x=np.concatenate((rx,-rx))
            y=np.concatenate((ry,-ry))
            i=i+1
        return x, y
    
def max_dataset_feature(dataset):
    max_feature = 0
    for data in dataset:
        max_feature = max(max_feature, torch.max(data.x).item())
    return max_feature

if __name__ == '__main__':
    dataset = EqualMomentsDatasets(root=os.path.join('DATA', "equal_moments"), k=1, num_pairs=100, 
                                   min_epsilon=0.01, max_epsilon=0.1, add_outliers=True)
    print(max_dataset_feature(dataset))
