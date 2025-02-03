from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn as nn
import torch










class RandomWalkEncoder(nn.Module):
    def __init__(self, rw_steps, pe_embed_dim, x_dim, embed_dim, attr_name='rw_pe', expand_x=False) -> None:
        super().__init__()
        self.attr_name = attr_name
        self.bn = nn.BatchNorm1d(rw_steps)
        self.linear = nn.Linear(rw_steps, pe_embed_dim)
        self.expand_x = expand_x
        if expand_x and embed_dim-pe_embed_dim > 0:
            self.linear_x = nn.Linear(x_dim, embed_dim-pe_embed_dim)
        elif expand_x:
            raise ValueError('embed_dim - pe_embed_dim should be greater than 0 if expand_x is True')
    
    def forward(self, batch):
        x = batch['x']
        if self.expand_x:
            x = self.linear_x(x)
        pe = self.linear(self.bn(batch[self.attr_name]))
        batch['x'] = torch.cat([x, pe], dim=-1)
        return batch
    

class LaplacianEigenvectorEncoder(nn.Module):
    def __init__(self, k, pe_embed_dim, x_dim, embed_dim, attr_name='lape_pe', expand_x=False) -> None:
        super().__init__()
        self.attr_name = attr_name
        self.mlp = nn.Sequential(
            nn.Linear(k, pe_embed_dim),
            nn.ReLU(),
            nn.Linear(pe_embed_dim, pe_embed_dim)
        )
        self.expand_x = expand_x
        if expand_x and embed_dim-pe_embed_dim > 0:
            self.linear_x = nn.Linear(x_dim, embed_dim-pe_embed_dim)
        elif expand_x:
            raise ValueError('embed_dim - k should be greater than 0 if expand_x is True')
    
    def forward(self, batch):
        x = batch['x']
        if self.expand_x:
            x = self.linear_x(x)
        pe = self.mlp(batch[self.attr_name])
        # pe = self.linear(self.bn(batch[self.attr_name]))
        batch['x'] = torch.cat([x, pe], dim=-1)
        return batch


class ComposeNodeEncoders(nn.Module):
        def __init__(self, encoder_names, encoder_kwargs) -> None:
            super().__init__()
            # check only two encoders are used, one for atom and one for positional encoding
            if len(encoder_names) > 2:
                raise ValueError('only two encoders are supported, one for atom and one for positional encoding')
            elif len(encoder_names) == 2 and ('atom' not in encoder_names or encoder_names[0] == 'atom' and encoder_names[1] == 'atom'):
                raise ValueError('can only use two encoders if one of them is atom encoder, and the other is positional encoder')
            
            self.encoders = nn.ModuleList()
            self.encoder_types = encoder_names
            x_dim = encoder_kwargs['in_dim']
            for name in encoder_names:
                if name == 'atom':
                    atom_embed_dim = encoder_kwargs['embed_dim']
                    if len(encoder_names) > 1:
                        atom_embed_dim -= encoder_kwargs['pe_embed_dim']
                    self.encoders.append(AtomEncoder(atom_embed_dim))
                    x_dim = encoder_kwargs['embed_dim']

                elif name == 'rwse':
                    self.encoders.append(RandomWalkEncoder(encoder_kwargs['rw_steps'], encoder_kwargs['pe_embed_dim'], 
                                                           x_dim, encoder_kwargs['embed_dim']))
                elif name == 'lape':
                    self.encoders.append(LaplacianEigenvectorEncoder(encoder_kwargs['lape_k'], encoder_kwargs['pe_embed_dim'], 
                                                           x_dim, encoder_kwargs['embed_dim']))
                else:
                    raise NotImplementedError('encoders available are [atom|rwse|lape]')
            
        
        def forward(self, batch):
            for i, encoder in enumerate(self.encoders):
                if self.encoder_types[i] == 'atom':
                    batch.x = encoder(batch['x'])
                else:
                    batch = encoder(batch)

            return batch


def setup_node_encoder(args):
    encoder_names = args.positional_encoding if args is not None else None
    if encoder_names not in ['atom', 'rwse', 'lape', 'atom+rwse', 'atom+lape', None, 'none']:
        raise NotImplementedError('encoders available are [atom|rwse|lape|atom+rwse|atom+lape]')
    if encoder_names is None or 'none':
        return lambda x: x

    encoder_names = encoder_names.split('+')
    return ComposeNodeEncoders(encoder_names, vars(args))



def setup_edge_encoder(dim):
    return BondEncoder(dim)




# test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    encoder_kwargs = {
        'embed_dim':128,
        'rw_steps':20,
        'lape_k':10,
        'encoder_names':'atom+lape',
        'pe_embed_dim':28,
        'in_dim':28
    }

    args = parser.parse_args([])
    for key, value in encoder_kwargs.items():
        setattr(args, key, value)

    encoder = setup_node_encoder(args)
    print(encoder)



    


    
