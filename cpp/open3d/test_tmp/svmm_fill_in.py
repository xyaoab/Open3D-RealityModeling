import torch
from torch_sparse import spmm

torch.manual_seed(0)

# param: 4 location, (128, 3) linear layers
C_in = 3
C_out = 128

N_param = 4
N_input = 10

params = torch.rand(N_param, C_out, C_in)

# input: (3, 10)
input = torch.rand(C_in, N_input)
index = torch.randint(high=N_param, size=(N_input,))

# Transpose both params (dense) and input (sparse) for spmm: sparse x dense
param_transposed = params.permute(0, 2, 1).reshape(N_param * C_in, C_out)
param_transposed.requires_grad = True

# Generate fill-in
ind_r = []
ind_c = []
val = []
# TBD: vectorize
for i in range(input.size(1)):
    ind_r += [i for k in range(C_in)]
    ind_c += list(range(index[i] * C_in, (index[i] + 1) * C_in))
    val.append(input[:, i])

ind = torch.stack((torch.tensor(ind_r), torch.tensor(ind_c)))
val = torch.cat(val)

print(ind)
print(val)
out = spmm(ind, val, N_input, C_in * N_param, param_transposed)
print(out)
print(params[index[0]] @ input[:,0])

out.sum().backward()



