import numpy as np

try:
  NP_DTYPE = np.float128
except:
  NP_DTYPE = np.float64


def approximate_lower_exponent(x, y, alpha_low, alpha_high, step_size, p=0.5, num_lower_envelopes=1, normalize=True):
  """
  Given data (x,y) s.t.: C*x_i + n>= y_i >= c*x_i^{\alpha} + n,
  where n is noise, this function approximates which of np.linspace(alpha_low, alpha_high, step_size) is the most probable alpha.
  :param: x - an array of x values.
  :param: y - an array of y values cooresponding to the x values.
  :param: alpha_low - minimal alpha to consider
  :param: alpha_high - maximal alpha to consider
  :param: step_size - step size between each pair of alphas to consider
  :param: p - for every considered alpha, the quality of the approximation is computed as \sum_i (x_i^alpha-y_i)^p.
          p must be less than 1.
  :param: num_lower_envelopes - number of lower envelopes of points to use for exponent approximation
  :param: normalize - whether to normalize penalty per point by point value
  :returns: alpha:float - approximate exponent alpha from data distribution
  :returns: c:float - approximate multiplier c from data distribution
  """
  assert(alpha_high >= alpha_low), "alpha high cannot be smaller than alpha_low"
  assert(p <= 1), "p must be less than or equal to 1"
  # prepare data
  x=np.array(x, dtype=NP_DTYPE)
  y=np.array(y, dtype=NP_DTYPE)
  # remove points where input distance is zero
  y=y[x!=0]
  x=x[x!=0]

  y=y[len(x)//2:]
  x=x[len(x)//2:]

  ratios = np.divide(y, x)
  rankings = np.argsort(ratios)
  used = np.zeros_like(rankings)

  for _ in range(num_lower_envelopes):
    curr_x = np.min(x) - 1

    for i in rankings:
      if used[i] == 0 and x[i] > curr_x:
        used[i] = 1
        curr_x = x[i]

  x = x[used==1]
  y = y[used==1]

  # compute num samples
  num_samples = int((alpha_high - alpha_low)//step_size) + 1
  best_penalty = float('inf')
  best_alpha = -1
  best_c = 0

  if alpha_low == alpha_high:
    y = np.power(y,1/alpha_low)
    best_c = np.min(np.power(y/x,alpha_low))
    best_alpha = alpha_low

  else:
    for alpha in np.linspace(alpha_low, alpha_high, num_samples):
      y_tmp = np.power(y, 1/alpha)
      c_holder=np.min(np.power(np.divide(y_tmp, x), alpha))
      approximated_y = c_holder*np.power(x,alpha)
      if normalize:
        penalty = np.sum(np.power(np.abs((y-approximated_y)/y), p))
      else:
        penalty = np.sum(np.power(np.abs(y-approximated_y), p))
      if penalty < best_penalty:
        best_penalty = penalty
        best_alpha = alpha
        best_c = c_holder


  return best_alpha, best_c



def approximate_upper_lipschitz_bound(x, y):
  """
  Given data (x,y) s.t.: C*x_i>= y_i,
  where n is noise, this function approximates C.
  :param: x - an array of x values.
  :param: y - an array of y values cooresponding to the x values.
  :returns: c:float - approximate multiplier c from data distribution
  """
  # prepare data
  x=np.array(x, dtype=NP_DTYPE)
  y=np.array(y, dtype=NP_DTYPE)
  y=y[x!=0]
  x=x[x!=0]

  

  ratios = np.divide(y, x)

  C = np.max(ratios)
  return C




holder_exponent_dict = {
    'relu': lambda x: (x+1)/x,
    'adaptive_relu': lambda x: 1,
    'sort': lambda x: 1,
    'sigmoid': lambda x: 2
}

def compute_theoretical_lower_holder_exponent(aggregation, height, max_degree, max_nodes, p=2):
  """
  Computes theoretically proven lower Holder exponent, assuming all aggregations are of the same type, and readout uses the same aggregation scheme.
  :param: aggregation - typre of aggregation function
  :param: height - height of epsilon tree
  :param: max_degree - max degree of any node across all dataset
  :param: max_nodes - max nodes per graph across all graphs in dataset
  :param: p - p for which we are using L_p distances

  :return: exponent -  theoretically proven lower Holder exponent
  """
  if aggregation not in holder_exponent_dict.keys():
    raise NotImplementedError(f'aggregation must be one of {holder_exponent_dict.keys()}')
  
  embedding_exponent = holder_exponent_dict[aggregation](p)

  if aggregation == 'relu':
    return 1 + (height-1) / p
  
  else:
    return embedding_exponent * (embedding_exponent**(height-2))

  # if aggregation == 'smooth':
  #   single_layer_exponent = holder_exponent_dict[aggregation](p)
  #   readout_exponent = holder_exponent_dict[aggregation](p)

  # else:
  #   single_layer_exponent = holder_exponent_dict[aggregation](p)
  #   readout_exponent = holder_exponent_dict[aggregation](p)

  # return readout_exponent * (single_layer_exponent**(height-2))
