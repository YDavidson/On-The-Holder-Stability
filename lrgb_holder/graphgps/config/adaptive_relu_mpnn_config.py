from torch_geometric.graphgym.register import register_config


@register_config('adaptive_relu_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True
    
    
    cfg.gnn.norm = 'batch'

    cfg.gnn.combine = 'ConcatProject'
    cfg.gnn.add_sum = True
    cfg.gnn.clamp_convex_weight = True
    cfg.gnn.linspace_bias = False
    cfg.gnn.pooling = 'adaptive_relu_global'

