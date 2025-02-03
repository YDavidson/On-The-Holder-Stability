import argparse
import os
import json
import torch
import os
import argparse
import json
from losses import setup_loss
from data_handling import setup_dataloaders
from models import setup_model
from optimizers import setup_optimizer
from trainer.trainer import Trainer





def test(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # get data
    train_loader, val_loader, test_loader = setup_dataloaders(args)
    # get model
    model = setup_model(args)
    # get optimizer
    optimizer = setup_optimizer(model, args)
    # get criterion
    criterion = setup_loss(args)
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get trainer
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, 
                      args.summary_path, args.task_level)
    
    
    # test best model
    trainer.test(test_loader)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary_path', type=str)
    args = parser.parse_args()

    # load args from summary path
    with open(os.path.join(args.summary_path, 'config.json'), 'r') as f:
        args = json.load(f)

    