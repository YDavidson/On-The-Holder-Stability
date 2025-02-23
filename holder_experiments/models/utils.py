from torch_geometric.nn.norm import LayerNorm, BatchNorm
import torch



def compute_model_size(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))


norm_dict = {'layer': LayerNorm, 'batch': BatchNorm, None: torch.nn.Identity}