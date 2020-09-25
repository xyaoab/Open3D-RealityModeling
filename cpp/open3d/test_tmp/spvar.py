import torch
from torch.utils import dlpack
import torch.nn as nn
import open3d as o3d
import numpy as np
from tqdm import tqdm


class SpVarModel(nn.Module):

    def __init__(self, device=o3d.core.Device('cpu:0')):
        super(SpVarModel, self).__init__()

        self.device = device

        coords = o3d.core.Tensor([[0], [1]],
                                 dtype=o3d.core.Dtype.Int64,
                                 device=device)
        elems = o3d.core.Tensor([[0.], [0.]],
                                dtype=o3d.core.Dtype.Float32,
                                device=device)
        self.spvar_params = o3d.core.SparseTensor(coords, elems)
        iterators, masks = self.spvar_params.insert_entries(coords, elems)
        params = self.spvar_params.get_elems_list(iterators[masks])

        self.coords = [0, 1]
        self.params = nn.ParameterList([nn.Parameter(dlpack.from_dlpack(param.to_dlpack()))
                                        for param in params])


    def forward(self, cs, xs):
        # Replace this with parallel hashmap.find(coords)
        out = []
        for c, x in zip(cs, xs):
            i = self.coords.index(c.item())
            out.append(self.params[i])
        return torch.stack(out).squeeze()

    def add_param(self, coords, device, optim):
        # Replace this with hashmap.activate(coords)
        # iterators, masks = self.hashmap.activate(coords)
        # SparseTensor(iterators[masks]) => list of tensors

        for coord in coords:
            if not coord in self.coords:
                param = nn.Parameter(nn.Parameter(torch.randn(1).to(device)))

                self.coords.append(coord.item())
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
    batchsize = 10
    spatial_slots = 10
    torch.manual_seed(0)
    np.random.seed(0)

    coords = torch.randint(spatial_slots, (n,)).to(device)
    xs = torch.randn(n).to(device)
    gts = (coords.float() / spatial_slots).to(device)

    for epoch in tqdm(range(1000)):
        for b in range(0, n, batchsize):
            optim.zero_grad()

            coords_b = coords[b:b + batchsize]
            xs_b = xs[b:b + batchsize]
            gts_b = gts[b:b + batchsize]

            model.add_param(coords_b, device, optim)

            out = model(coords_b, xs_b)
            loss = (gts_b - out).norm()
            loss.backward()
            optim.step()

    for c, p in zip(model.coords, model.params):
        print(c, p)
