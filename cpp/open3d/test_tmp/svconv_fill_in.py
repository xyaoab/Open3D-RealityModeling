import torch
from torch_sparse import spmm

torch.manual_seed(0)

# param: 4 location, (128, 3) linear layers
C_in = 3
C_out = 6
kernelsize = 7

N_param = 4
N_input = 10

params = torch.rand(N_param, kernelsize, C_out, C_in)

# We'd better store them in this way, permuted
param_transposed = params.permute(0, 1, 3, 2).reshape(N_param * kernelsize * C_in, C_out)

# input: (3, 10)
input = torch.rand(C_in, N_input)
index = torch.randint(high=N_param, size=(N_input,))
print(index)
nb_indices = torch.randint(low=-1, high=N_input, size=(N_input, kernelsize))
nb_masks = nb_indices >= 0

ind_r = []
ind_c = []
val = []
nbs = torch.arange(kernelsize)

for i in range(input.size(1)):
    print(i)

    sp_ind = index[i]
    sp_offset = sp_ind * kernelsize * C_in

    valid_nb_inds_in_global = nb_indices[i][nb_masks[i]]
    valid_nb_inds_in_local = nbs[nb_masks[i]]
    num_valid_nbs = len(valid_nb_inds_in_local)

    local_r = [i for k in range(C_in * num_valid_nbs)]
    local_c = []
    for nb_ind in valid_nb_inds_in_local:
        local_c += list(
            range(sp_offset + nb_ind * C_in, sp_offset + nb_ind * C_in + C_in))
    local_v = [input[:, nb_ind] for nb_ind in valid_nb_inds_in_global]

    ind_r += local_r
    ind_c += local_c
    val += local_v

ind = torch.stack((torch.tensor(ind_r), torch.tensor(ind_c)))
val = torch.cat(val)

print(ind)
print(val)

out = spmm(ind, val, N_input, C_in * kernelsize * N_param, param_transposed)
