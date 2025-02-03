from torch.optim import Adam, SGD, RMSprop, AdamW


def setup_optimizer(model, args):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}, must be one of {}'.format(args.optimizer, ['adam', 'sgd', 'rmsprop']))