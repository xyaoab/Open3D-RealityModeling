import torch
from torch.utils import dlpack
import torch.nn as nn
import open3d as o3d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SpVarModel(nn.Module):

    def __init__(self, device=o3d.core.Device('cpu:0')):
        super(SpVarModel, self).__init__()

        self.device = device

        coords = o3d.core.Tensor([[0]],
                                 dtype=o3d.core.Dtype.Int64,
                                 device=device)
        elems = o3d.core.Tensor([[0.]],
                                dtype=o3d.core.Dtype.Float32,
                                device=device)
        indices = o3d.core.Tensor([[0]],
                                  dtype=o3d.core.Dtype.Int64,
                                  device=device)

        self.spvar_params = o3d.core.SparseTensor(coords, elems)
        iterators, masks = self.spvar_params.insert_entries(coords, elems)
        params = self.spvar_params.get_elems_list(iterators[masks])

        self.coord_param_map = o3d.core.SparseTensor(coords,
                                                     indices,
                                                     insert=True)
        self.params = nn.ParameterList([
            nn.Parameter(dlpack.from_dlpack(param.to_dlpack()))
            for param in params
        ])

    def forward(self, cs, xs):
        # Replace this with parallel hashmap.find(coords)
        o3d_coords = o3d.core.Tensor.from_dlpack(dlpack.to_dlpack(cs))
        iterators, masks = self.coord_param_map.find_entries(o3d_coords)

        elem_list = self.coord_param_map.get_elems_list(iterators[masks])

        out = []
        for i, elem in enumerate(elem_list):
            out.append(self.params[elem[0].item()] * xs[i])
        return torch.stack(out)

    def add_param(self, coords, device, optim):
        o3d_coords = o3d.core.Tensor.from_dlpack(dlpack.to_dlpack(coords))
        iterators, masks = self.spvar_params.activate_entries(o3d_coords)

        new_params = self.spvar_params.get_elems_list(iterators[masks])

        if len(new_params) == 0:
            return

        self.coord_param_map.insert_entries(
            o3d_coords[masks],
            o3d.core.Tensor(np.expand_dims(np.arange(
                len(self.params),
                len(self.params) + len(new_params)),
                                           axis=1),
                            dtype=o3d.core.Dtype.Int64))

        for i, param in enumerate(new_params):
            param = nn.Parameter(dlpack.from_dlpack(param.to_dlpack()))
            self.params.append(param)
            optim.add_param_group({'params': param})


if __name__ == '__main__':
    device = 'cpu'

    coords = o3d.core.Tensor([[100], [200], [300]], dtype=o3d.core.Dtype.Int64)
    elems = o3d.core.Tensor([[1.0], [2.0], [3.0]], dtype=o3d.core.Dtype.Float32)
    sp_tensor = o3d.core.SparseTensor(coords, elems, insert=True)
    iterators, masks = sp_tensor.find_entries(coords)
    sp_tensor_elem_list = sp_tensor.get_elems_list(iterators[masks])

    model = SpVarModel().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Generate test data:
    # > param(coord) = coord * 0.01
    # > gt: param(coord) * x(coord)
    n = 1000
    batchsize = 1000
    spatial_slots = 10
    torch.manual_seed(0)
    np.random.seed(0)

    coords = torch.randint(spatial_slots, (n,)).to(device)
    xs = torch.randn(n).to(device)
    gts = (coords.float() / spatial_slots).to(device) * xs

    losses = []
    for epoch in tqdm(range(1000)):
        for b in range(0, n, batchsize):
            optim.zero_grad()

            coords_b = coords[b:b + batchsize].unsqueeze(1)
            xs_b = xs[b:b + batchsize].unsqueeze(1)
            gts_b = gts[b:b + batchsize].unsqueeze(1)

            model.add_param(coords_b, device, optim)

            out = model(coords_b, xs_b)
            # print(out.size(), gts_b.size())
            loss = (gts_b - out).norm()
            loss.backward()
            losses.append(loss.item())
            optim.step()

    coords = o3d.core.Tensor(np.expand_dims(np.arange(0, 10), axis=1),
                             dtype=o3d.core.Dtype.Int64)
    iterators, masks = model.spvar_params.find_entries(coords)
    elems_list = model.spvar_params.get_elems_list(iterators[masks])
    for elem in elems_list:
        print(elem)

    plt.plot(np.arange(0, len(losses)), np.stack(losses).T)
    plt.show()
