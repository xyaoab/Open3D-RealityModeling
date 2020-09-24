import torch
from torch.utils import dlpack
import torch.nn as nn
import open3d as o3d


class SpVarModel(nn.Module):

    def __init__(self):
        super(SpVarModel, self).__init__()

        # Default 1d coordinate 0, to be extended
        self.coords = [0]
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(1))
        ])

    def forward(self, cs, xs):
        # Replace this with parallel hashmap.find(coords)
        out = []
        for c, x in zip(cs, xs):
            i = self.coords.index(c)
            out.append(self.params[i] * x)
        return torch.stack(out)

    def add_param(self, coords, device, optim):
        # Replace this with hashmap.activate(coords)
        for coord in coords:
            if not coord in self.coords:
                param = nn.Parameter(nn.Parameter(torch.randn(1).to(device)))

                self.coords.append(coord)
                self.params.append(param)
                optim.add_param_group({'params': param})


if __name__ == '__main__':
    device = 'cpu'

    model = SpVarModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    # Generate test data:
    # > param(coord) = coord * 0.01
    # > gt: param(coord) * x(coord)
    n = 10000
    batchsize = 1
    spatial_slots = 100

    coords = torch.randint(spatial_slots, (n,)).to(device)
    xs = torch.randn(n).to(device)
    gts = (coords.float() / spatial_slots * xs).to(device)

    for epoch in range(1000):
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

        print(epoch, loss)

    for i, param in enumerate(model.params):
        print(i, param.item())
