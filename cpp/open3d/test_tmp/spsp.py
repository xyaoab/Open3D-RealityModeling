import torch
from torch_sparse import spspmm

# Linear layer
# params: (N, (128, 3))
# inputs: (M, (3,))

# Prepare params
param0 = torch.randn(5, 3).flatten()
param1 = torch.randn(5, 3).flatten()
param2 = torch.randn(5, 3).flatten()

param_value = torch.cat((param0, param1, param2))

index_i, index_j = torch.meshgrid(torch.arange(5), torch.arange(3))
param0_index = torch.stack((index_i.flatten(), index_j.flatten()))
param1_index = torch.stack(((index_i + 5).flatten(), (index_j + 3).flatten()))
param2_index = torch.stack(((index_i + 5 * 2).flatten(), (index_j + 3 * 2).flatten()))

param_index = torch.cat((param0_index, param1_index, param2_index), axis=1)

# Prepare inputs
input0 = torch.randn(3, 10).flatten()
input1 = torch.randn(3, 1).flatten()
input2 = torch.randn(3, 7).flatten()

input_value = torch.cat((input0, input1, input2))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(10))
input0_index = torch.stack((index_i.flatten(), index_j.flatten()))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(1))
input1_index = torch.stack(((index_i + 3).flatten(), (index_j + 10).flatten()))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(7))
input2_index = torch.stack(((index_i + 3 * 2).flatten(), (index_j + 10 + 1).flatten()))

input_index = torch.cat((input0_index, input1_index, input2_index), axis=1)

output_index, output_value = spspmm(param_index, param_value, input_index, input_value, 5 * 3, 3 * 3, 10 + 1 + 7)
torch.set_printoptions(profile="full")
print(param_index)
print(input_index)

print(output_index)
