import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import scipy
from scipy import optimize
from numpy import *
from numpy import trace
from numpy import linalg as LA
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os



def relu(X):
  X=torch.tensor(X)
  Y=torch.relu(X)
  return(Y.numpy())

X=np.random.randn(3,3)
Y=relu(X)
print(X)
print(Y)

class MLPSum(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(MLPSum, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation()
        )

        torch.nn.init.xavier_normal_(self.MLP[0].weight)
        # Normalize weights so they are on the unit sphere
        with torch.no_grad():
          self.MLP[0].weight /= torch.norm(self.MLP[0].weight, dim=1, keepdim=True)
          torch.nn.init.uniform_(self.MLP[0].bias, -2, 2)

    def forward(self, x):
        """
        :param: x [num_instances, feature_dim].
        """
        return torch.sum(self.MLP(x), dim=0)


class SortProject(nn.Module):
    def __init__(self, in_dim, out_dim, num_elements):
        super(SortProject, self).__init__()

        self.lin_project = torch.nn.Linear(in_dim, out_dim, bias=False)

        self.lin_collapse = torch.nn.Parameter(torch.zeros((num_elements,out_dim)))

        torch.nn.init.xavier_normal_(self.lin_project.weight)
        torch.nn.init.normal_(self.lin_collapse)

    def forward(self, x):
        """
        :param: x [num_instances, feature_dim].
        """
        p = self.lin_project(x)
        return torch.sum(p*self.lin_collapse, dim=1)


class AdaptiveReLU(nn.Module):
    def __init__(self, feature_dim):
        """
        Implementation of adaptive bias relu.
        Per multiset X, bias is chosen as t*max(<a,X>) + 1-t(<a,X>),
        where t\in [0,1] is a learned parameter.
        The function maps a multiset {{x_1,...,x_n}}->[n, Max, Min, \sum ReLU(x_i-t*Max-(1-t)*Min)],
        and then proceeds to project the 4 coordinates to a single scalar.
        This is done for each feature dimension separately.
        """
        super(AdaptiveReLU, self).__init__()

        self.feature_dim = feature_dim
        # differentiable function to count number of elements in each multiset
        self.bincount = lambda inds, arr: torch.scatter_reduce(input=arr, dim=0, index=inds,
                                                               src=torch.ones_like(inds, device=arr.device), reduce="sum")

        self.scalars = nn.Parameter(torch.zeros([4,feature_dim]).normal_(mean=0,std=1))
        self.t = nn.Parameter(torch.FloatTensor(feature_dim).uniform_(0,1))

    def forward(self, x):
        """
        :param: x [num_instances, feature_dim].
        """
        batch_idx = torch.zeros(x.shape[0], dtype=torch.int64)

        num_nodes = x.shape[0]+1

        # Compute min and max per batch_idx for each feature dimension
        min_values = scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='min')
        repeated_min_values = torch.zeros_like(x)
        repeated_min_values += min_values[batch_idx,:]

        max_values = scatter(x, batch_idx, dim=0, dim_size=num_nodes, reduce='max')
        repeated_max_values = torch.zeros_like(x)
        repeated_max_values += max_values[batch_idx,:]

        # Compute bias as a convex combination of mins and maxs using t
        bias = self.t * repeated_max_values + (1 - self.t) * repeated_min_values

        # Add output of linear layer with bias
        translated = x - bias

        # Perform elementwise max(0, ...)
        post_relu = F.relu(translated)
        relu_sum = scatter(post_relu, batch_idx, dim=0, dim_size=num_nodes, reduce='sum')

        # compute number of elements per multiset
        num_elements = self.bincount(batch_idx,
                                     torch.zeros(num_nodes, dtype=torch.long, device=x.device)).unsqueeze(-1).repeat(1, self.feature_dim)

        output = self.scalars[0]*num_elements + self.scalars[1]*min_values + self.scalars[2]*max_values + self.scalars[3]*relu_sum

        return output.squeeze(-1)


def embed(X,A,embedding,B,b):
  n=X.shape[1]
  b_clone=np.outer(b,np.ones(n))
  match embedding:
        case 'sort':
          Z = np.dot(A, X)
          W = np.sort(Z, axis=1)
          Y=  np.sum(B*W,axis=1)
        case 'relu':
          Z=np.dot(A,X)+b_clone
          Z=relu(Z)
          Y=np.sum(Z,axis=1)
        case 'sigmoid':
          Z=np.dot(A,X)+b_clone
          Z=1/(1 + np.exp(-Z))
          Y=np.sum(Z,axis=1)
        case 'adaptive_relu':
          pass

  return Y



def dist(X1, X2):
    X3 = np.transpose(X1)
    M1 = np.dot(X3, X2)  # this part is for denominator
    Mt = np.transpose(M1)
    
    I, F = scipy.optimize.linear_sum_assignment(Mt, maximize=True)
    permutation_matrix = csr_matrix(
        (np.ones(n, dtype=int), (F, I)), shape=(n, n))
    P = permutation_matrix.todense()
    X1_s = LA.norm(X1)
    R = X1_s*X1_s
    X2_s = LA.norm(X2)
    S = X2_s*X2_s
    T1 = np.dot(P, Mt)
    T = matrix.trace(T1)
    D2 = R+S-2*T
    D2 = np.abs(D2)
    L = sqrt(D2)
    return L




#generate two distinct sorted vectors x,y of length 2^{k+1} whose first 2^k moments are the same
def arbitrary_moments(k):
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
  #check correctness: x and y have 2^k identical moments
  for j in range(1,len(x)+1):
    print(['printing moments of  order',j])
    mxj=np.sum(x**j)
    print(mxj)
    myj=np.sum(y**j)
    print(myj)
    print('difference is')
    print(np.linalg.norm(mxj-myj))
  return x,y



k=3
[x,y]=arbitrary_moments(k)
smooth_holder=2**(k+1)




torch.set_default_dtype(torch.float64)
#experiment parameters
pair_num=200
add_random=False #if True, some of the pairs in the scatter plot will not be adversarial
n=len(x)
d=1
x0=np.zeros(n)
D=10000
Seed=1
np.random.seed(Seed)
torch.manual_seed(Seed)
embeddings = [MLPSum(d,D,nn.Sigmoid), MLPSum(d,D,nn.ReLU), AdaptiveReLU(D), SortProject(d, D, n)]
embedding_names = ['sigmoid', 'relu', 'adaptive_relu', 'sort']
exponents=[smooth_holder,3/2, 1, 1]
embed_num=len(embeddings)

embed_dist=-np.ones([embed_num,pair_num])

t_vec=np.linspace(1,0,pair_num,endpoint=False) #vector of `small ts'
W2_dist=-np.ones(pair_num)
with torch.no_grad():
  for i in range(pair_num):
    t=t_vec[i]
    X1=x0+t*x
    X2=x0+t*y
    if add_random:
      if random.randint(0,2)==0:
        X1=0.1*np.random.randn(len(x))
        X2=0.1*np.random.randn(len(x))
    W2_dist[i]=np.linalg.norm(np.sort(X1)-np.sort(X2))
    for j in range(embed_num):
      Y1=embeddings[j](torch.tensor(X1).unsqueeze(-1))
      Y2=embeddings[j](torch.tensor(X2).unsqueeze(-1))
      embed_dist[j,i] =1/(np.sqrt(D))*LA.norm(np.subtract(Y1, Y2))



SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


ratio=-np.ones([embed_num,pair_num])
ratio_holder=-np.ones([embed_num,pair_num])
M=-np.ones(embed_num)
m=-np.ones(embed_num)
m_holder=-np.ones(embed_num)
for j in range(embed_num):
  holder_exponent = exponents[j]
  ratio[j,:]=embed_dist[j,:]/W2_dist
  ratio_holder[j,:]=embed_dist[j,:]/np.power(W2_dist,holder_exponent)
  M[j]=np.max(ratio[j])
  m[j]=np.min(ratio[j])
  m_holder[j]=np.min(ratio_holder[j])
print(M)
print(m)
print(m_holder)

W2max=np.max(W2_dist)
x=np.linspace(0,W2max,1000)
fig, axs = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)
fig.text(0.5, -0.03, r'$W_2$', ha='center')
fig.text(-0.03, 0.5, r'$L_2$', va='center', rotation='vertical')
for j in range(embed_num):
  ax = axs[j//2, j%2]
  ax.set_title(embedding_names[j])
  ax.plot(x,m_holder[j]*np.power(x,exponents[j]))
  ax.scatter(W2_dist,embed_dist[j,:], c='orange', alpha=1.0, marker='x')
  if j == 0:
    ax.legend([r'$\alpha$'+f'={exponents[j]}', 'datapoints'])
  else:
    ax.legend([r'$\alpha$'+f'={exponents[j]}'])
  
# create exp_plots directory if it does not exist
if not os.path.exists('exp_plots'):
    os.makedirs('exp_plots')

plt.savefig('exp_plots/multiset_exponents.png')

# clear the plot
plt.clf()


# relu vs smooth per width on n-1 equal moments example
k=2
[x,y]=arbitrary_moments(k)

torch.set_default_dtype(torch.float64)
#experiment parameters
n=len(x)
epsilon = 1/10
x0=np.zeros(n)
X1=x0+epsilon*x
X2=x0+epsilon*y

Wasserstein_dist = np.linalg.norm(np.sort(X1)-np.sort(X2), ord=1)

d=1

num_widths = 10
widths = np.logspace(0, 8, num=num_widths, base=2)
Seed=1
np.random.seed(Seed)
torch.manual_seed(Seed)
embed_dists = {'smooth': [], 'relu': []}

with torch.no_grad():
  for D in widths:
    D = int(D)
    relu_embedding = MLPSum(d,D,nn.ReLU)
    sigmoid_embedding = MLPSum(d,D,nn.Sigmoid)

    embed_dists['relu'].append(1/(np.sqrt(D))*LA.norm(np.subtract(relu_embedding(torch.tensor(X1).unsqueeze(-1)), relu_embedding(torch.tensor(X2).unsqueeze(-1)))))
    embed_dists['smooth'].append(1/(np.sqrt(D))*LA.norm(np.subtract(sigmoid_embedding(torch.tensor(X1).unsqueeze(-1)), sigmoid_embedding(torch.tensor(X2).unsqueeze(-1)))))

    
    # # W2_dist[i]=np.linalg.norm(np.sort(X1)-np.sort(X2))
    # for j in range(embed_num):
    #   Y1=embeddings[j](torch.tensor(X1).unsqueeze(-1))
    #   Y2=embeddings[j](torch.tensor(X2).unsqueeze(-1))
    #   embed_dist[j,i] =1/(np.sqrt(D))*LA.norm(np.subtract(Y1, Y2))

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.xlabel('Width')
plt.ylabel('Embedding distance')
plt.loglog(widths, embed_dists['relu'], label='relu')
plt.loglog(widths, embed_dists['smooth'], label='smooth')
plt.axhline(y=Wasserstein_dist, color='r', linestyle='--', label=r'$W_1$')

plt.yscale('symlog', linthresh=1e-17)
# Find the minimum non-zero value in your data
min_value = min(np.min(embed_dists['relu']), 
                np.min(embed_dists['smooth']))

plt.ylim(bottom=-1e-16)  
yticks = [10**i for i in range(0, -20, -3)] 
plt.yticks(yticks)
plt.legend()
plt.grid()
plt.savefig('exp_plots/relu_vs_smooth_per_depth.png')


# clear the plot
plt.clf()


k=2
[x,y]=arbitrary_moments(k)

torch.set_default_dtype(torch.float64)
#experiment parameters
n=len(x)
epsilon = 1/10
x0=np.zeros(n)
X1=x0+epsilon*x
X2=x0+epsilon*y

Wasserstein_dist = np.linalg.norm(np.sort(X1)-np.sort(X2), ord=1)

d=1

num_widths = 10
widths = np.logspace(0, 8, num=num_widths, base=2)
Seed=1
np.random.seed(Seed)
torch.manual_seed(Seed)
embed_dists = {'smooth': [], 'relu': []}

with torch.no_grad():
  for D in widths:
    D = int(D)
    relu_embedding = MLPSum(d,D,nn.ReLU)
    sigmoid_embedding = MLPSum(d,D,nn.Sigmoid)

    embed_dists['relu'].append(1/(np.sqrt(D))*LA.norm(np.subtract(relu_embedding(torch.tensor(X1).unsqueeze(-1)), relu_embedding(torch.tensor(X2).unsqueeze(-1)))))
    embed_dists['smooth'].append(1/(np.sqrt(D))*LA.norm(np.subtract(sigmoid_embedding(torch.tensor(X1).unsqueeze(-1)), sigmoid_embedding(torch.tensor(X2).unsqueeze(-1)))))


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.xlabel('Width')
plt.ylabel('Embedding distance')
plt.loglog(widths, embed_dists['relu'], label='relu')
plt.loglog(widths, embed_dists['smooth'], label='smooth')
plt.axhline(y=Wasserstein_dist, color='r', linestyle='--', label=r'$W_1$')

plt.yscale('symlog', linthresh=1e-17)
# Find the minimum non-zero value in your data
min_value = min(np.min(embed_dists['relu']),
                np.min(embed_dists['smooth']))

plt.ylim(bottom=-1e-16)
yticks = [10**i for i in range(0, -20, -3)]
plt.yticks(yticks)
plt.legend()
plt.grid()
plt.savefig('exp_plots/relu_vs_smooth_per_width.png')