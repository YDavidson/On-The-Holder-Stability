import torch 
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, LRGBDataset
from utils.utils import get_max_degree_and_nodes
from data_handling.epsilon_trees import EpsilonTrees
from data_handling.equal_moments import EqualMomentsDatasets
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE, AddRandomWalkPE
# get stratified folds for cross validation from sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import itertools
from tqdm import tqdm


def get_dataset_splits(args):
    dataset_name = args.dataset
    if dataset_name == 'mutag':
        dataset = TUDataset(root='DATA', name='MUTAG')
    elif dataset_name == 'proteins':
        dataset = TUDataset(root='DATA', name='PROTEINS')
    elif dataset_name == 'enzymes':
        dataset = TUDataset(root='DATA', name='ENZYMES')
    elif dataset_name == 'ptc':
        dataset = TUDataset(root='DATA', name='PTC_MR')
    elif dataset_name == 'nci1':
        dataset = TUDataset(root='DATA', name='NCI1')
    elif dataset_name == 'nci109':
        dataset = TUDataset(root='DATA', name='NCI109')
    elif dataset_name == 'epsilon_trees':
        dataset = EpsilonTrees(root=os.path.join('DATA','epsilon_trees'), num_pairs=args.num_pairs, height=args.height, feature_dimension=args.feature_dimension,
                                min_epsilon=args.min_epsilon, max_epsilon=args.max_epsilon, task=args.task)
        
        indices = list(range(args.num_pairs))
        train_size = int(args.num_pairs * 0.8)

        train_indices, val_test_indices = train_test_split(indices, train_size=train_size, random_state=args.seed)
        val_indices, test_indices = train_test_split(val_test_indices, train_size=int(len(val_test_indices)/2), random_state=args.seed)
        # split pairs between train, val and test
        train_indices = list(itertools.chain(*zip([2*i for i in train_indices], [2*i+1 for i in train_indices])))
        val_indices = list(itertools.chain(*zip([2*i for i in val_indices], [2*i+1 for i in val_indices])))
        test_indices = list(itertools.chain(*zip([2*i for i in test_indices], [2*i+1 for i in test_indices])))

        return [dataset[train_indices]], [dataset[val_indices]], [dataset[test_indices]]
    elif dataset_name == 'equal_moments':
        dataset = EqualMomentsDatasets(root='DATA', k=args.k, num_pairs=args.num_pairs, min_epsilon=args.min_epsilon, 
                                       max_epsilon=args.max_epsilon, add_outliers=args.add_outliers)

        indices = list(range(args.num_pairs))
        train_size = int(args.num_pairs * 0.8)
        train_indices, val_test_indices = train_test_split(indices, train_size=train_size, random_state=args.seed)
        val_indices, test_indices = train_test_split(val_test_indices, train_size=int(len(val_test_indices)/2), random_state=args.seed)
        # split pairs between train, val and test
        train_indices = list(itertools.chain(*zip([2*i for i in train_indices], [2*i+1 for i in train_indices])))
        val_indices = list(itertools.chain(*zip([2*i for i in val_indices], [2*i+1 for i in val_indices])))
        test_indices = list(itertools.chain(*zip([2*i for i in test_indices], [2*i+1 for i in test_indices])))

        return [dataset[train_indices]], [dataset[val_indices]], [dataset[test_indices]]
    else:
        raise ValueError('Unknown dataset: {}, must be one of {}'.format(dataset_name, 
                                                                         ['mutag', 'proteins', 'enzymes',
                                                                          'ptc', 'nci1', 'nci109', 'epsilon_trees']))
    
    
    train_sets = []
    val_sets = []

    if args.num_folds is None:    
        indices = list(range(len(dataset)))
        train_size = int(len(dataset) * 0.9)
        train_indices, val_test_indices = train_test_split(indices, train_size=train_size, stratify=dataset.data.y, random_state=args.seed)
        # use val_test as val and test as done in 'How Powerfull are Graph Neural Networks'        
        train_sets.append(dataset[train_indices])
        val_sets.append(dataset[val_test_indices])
    else:
        num_folds = args.num_folds
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
        for train_index, test_index in skf.split(torch.zeros_like(dataset.data.y), dataset.data.y):
            # use val as val and test as done in 'How Powerfull are Graph Neural Networks'  
            train_sets.append(dataset[train_index.tolist()])
            val_sets.append(dataset[test_index.tolist()])
            
    return train_sets, val_sets, [[]]*len(val_sets)





def setup_dataloaders(args):
    train_sets, val_sets, test_sets = get_dataset_splits(args)
    dataset = train_sets[0] + val_sets[0] + test_sets[0]
    # get max num nodes and max num neigbors in dataset
    if args.model == 'sort_mpnn':
        max_neighbors, max_nodes = get_max_degree_and_nodes(dataset=dataset)
        args.max_neighbors = max_neighbors
        args.max_nodes = max_nodes

    shuffle_train = not args.no_shuffle_train
    train_loaders = [DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle_train, num_workers=args.num_workers) for train_set in train_sets]
    val_loaders = [DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) for val_set in val_sets]
    test_loaders = [DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) for test_set in test_sets]
    
    return train_loaders, val_loaders, test_loaders