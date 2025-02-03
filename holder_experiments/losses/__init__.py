import torch
import torch.nn as nn



def setup_loss(args):
    # regression losses
    if args.loss == 'mse':
        return nn.MSELoss()
    elif args.loss == 'mae':
        return nn.L1Loss()
    # classification losses
    elif args.loss == 'bce':
        return nn.BCELoss()
    elif args.loss == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss: {}, must be one of {}'.format(args.loss, ['mse', 'mae',
                                                                                  'bce', 'bce_with_logits',
                                                                                  'ce']))