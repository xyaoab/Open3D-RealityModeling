import torch
from torch.utils import dlpack
import torch.nn as nn
import open3d as o3d
from tqdm import tqdm

class SpVarModel(nn.Module):

    def __init__(self):
        super(SpVarModel, self).__init__()

        # Default 1d coordinate 0, to be extended
        self.coords = [i for i in range(10)]
        self.params = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1).uniform_(0, 1)) for i in range(10)
        ])

    def forward(self, cs, xs):
        # Replace this with parallel hashmap.find(coords)
        out = []
        for c, x in zip(cs, xs):
            i = self.coords.index(c.item())
            out.append(self.params[i])
        return torch.stack(out).squeeze()

    def add_param(self, coords, device, optim):
        # Replace this with hashmap.activate(coords)
        for coord in coords:
            if not coord in self.coords:
                param = nn.Parameter(nn.Parameter(torch.randn(1).to(device)))

                self.coords.append(coord.item())
                self.params.append(param)
                optim.add_param_group({'params': param})


if __name__ == '__main__':
    device = 'cpu'

    model = SpVarModel().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Generate test data:
    # > param(coord) = coord * 0.01
    # > gt: param(coord) * x(coord)
    n = 1000
    batchsize = 1000
    spatial_slots = 10
    torch.manual_seed(0)

    coords = torch.randint(spatial_slots, (n,)).to(device)
    xs = torch.randn(n).to(device)
    gts = (coords.float() / spatial_slots).to(device)
    print(coords)

    for epoch in tqdm(range(10000)):
        for b in range(0, n, batchsize):
            optim.zero_grad()

            coords_b = coords[b:b + batchsize]
            xs_b = xs[b:b + batchsize]
            gts_b = gts[b:b + batchsize]

            # model.add_param(coords_b, device, optim)

            out = model(coords_b, xs_b)
            loss = (gts_b - out).norm()
            loss.backward()
            print(loss.item())
            optim.step()


    for c, p in zip(model.coords, model.params):
        print(c, p)
