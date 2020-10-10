import torch
from torch_sparse import spspmm
from torch_sparse import spmm

# sp x vector
index = torch.tensor([[0, 0, 1, 2, 2],
                      [0, 2, 1, 0, 1]])
value = torch.Tensor([1, 2, 4, 1, 3])
value.requires_grad=True
matrix = torch.Tensor([[1, 4], [2, 5], [3, 6]])

out = spmm(index, value, 3, 3, matrix)
out.sum().backward()
print('spmm simple backward works')

# sp x sp
indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.Tensor([1, 2, 3, 4, 5])
valueA.requires_grad=True

indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.Tensor([2, 4])

indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)
valueC.sum().backward()
print('spspmm simple backward works')

# set up blocks
# Linear layer
# params: (N, (128, 3))
# inputs: (M, (3,))

# Prepare params
param0 = torch.randn(5, 3, requires_grad=True).flatten()
param1 = torch.randn(5, 3, requires_grad=True).flatten()
param2 = torch.randn(5, 3, requires_grad=True).flatten()

param_value = torch.cat((param0, param1, param2))

index_i, index_j = torch.meshgrid(torch.arange(5), torch.arange(3))
param0_index = torch.stack((index_i.flatten(), index_j.flatten()))
param1_index = torch.stack(((index_i + 5).flatten(), (index_j + 3).flatten()))
param2_index = torch.stack(
    ((index_i + 5 * 2).flatten(), (index_j + 3 * 2).flatten()))

param_index = torch.cat((param0_index, param1_index, param2_index), axis=1)

# Prepare inputs
input0 = torch.randn(3, 10, requires_grad=True).flatten()
input1 = torch.randn(3, 1, requires_grad=True).flatten()
input2 = torch.randn(3, 7, requires_grad=True).flatten()

input_value = torch.cat((input0, input1, input2))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(10))
input0_index = torch.stack((index_i.flatten(), index_j.flatten()))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(1))
input1_index = torch.stack(((index_i + 3).flatten(), (index_j + 10).flatten()))

index_i, index_j = torch.meshgrid(torch.arange(3), torch.arange(7))
input2_index = torch.stack(
    ((index_i + 3 * 2).flatten(), (index_j + 10 + 1).flatten()))

input_index = torch.cat((input0_index, input1_index, input2_index), axis=1)

output_index, output_value = spspmm(param_index, param_value, input_index,
                                    input_value, 5 * 3, 3 * 3, 10 + 1 + 7)
output_value.sum().backward()
