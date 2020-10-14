import torch
import open3d as o3d
import open3d.core as o3c
from torch_sparse import spmm
from torch.utils import dlpack

torch.manual_seed(0)

# param: 4 location, (128, 3) linear layers
C_in = 3
C_out = 6
kernelsize = 8

N_param = 4
N_input = 100

params = torch.rand(N_param, kernelsize, C_out, C_in)

# We'd better store them in this way, permuted
param_transposed = params.permute(0, 1, 3,
                                  2).reshape(N_param * kernelsize * C_in, C_out)
param_transposed.requires_grad=True

# input: (3, 10)
input = torch.rand(C_in, N_input)
input_discretized = (input / 0.4).long().T

index = torch.randint(high=N_param, size=(N_input,))

dtype_key = o3d.core.Dtype(o3d.core.DtypeCode.Object, 8 * 3, 'Int64x3')
hashmap = o3c.Hashmap(init_capacity=N_input,
                      dtype_key=dtype_key,
                      dtype_value=o3c.Dtype.Int64,
                      device=o3c.Device("cpu:0"))

input_discretized_o3d = o3c.Tensor.from_dlpack(
    dlpack.to_dlpack(input_discretized))
_, masks, indices = hashmap.activate(input_discretized_o3d)
key_tensor = hashmap.get_key_blob_as_tensor((N_input, 3), o3c.Dtype.Int64)
# print(indices[masks])
# print(key_tensor)

valid_entries = key_tensor[indices[masks].to(o3c.Dtype.Int64)]
N_valid = valid_entries.shape[0]
valid_entries_nbs = o3c.Tensor.empty((N_valid * 8, 3),
                                     dtype=valid_entries.dtype)
for i in range(8):
    offset = o3c.Tensor([(i & 1) > 0, (i & 2) > 0, (i & 4) > 0]).to(o3c.Dtype.Int64)
    valid_entries_nbs[N_valid * i : N_valid * (i + 1)] = offset + valid_entries
# print(valid_entries)
print(valid_entries_nbs[:N_valid])
_, masks, indices = hashmap.find(valid_entries_nbs)
indices = dlpack.from_dlpack(indices.to_dlpack()).reshape((N_valid, 8))

key_tensor_torch = dlpack.from_dlpack(key_tensor.to_dlpack())
nb_indices = indices
print(indices)
nb_masks = nb_indices >= 0

ind_r = []
ind_c = []
val = []
nbs = torch.arange(kernelsize)

for i in range(N_valid):
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
    local_v = [key_tensor_torch[nb_ind, :] for nb_ind in valid_nb_inds_in_global]

    ind_r += local_r
    ind_c += local_c
    val += local_v

ind = torch.stack((torch.tensor(ind_r), torch.tensor(ind_c)))
val = torch.cat(val)

print(ind)
print(val)

out = spmm(ind, val, N_valid, C_in * kernelsize * N_param, param_transposed)
print(out)
out.sum().backward()

