import torch
import torch.nn as nn
import open3d as o3d

class SpVarModel(nn.Module):
    def __init__(self):
        super(SpVarModel, self).__init__()

        self.coords = [0]
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1))])

    def forward(self, coord):
        i = self.coords.index(coord)
        return self.params[i]

    def add_param(self, coord, optim):
        if not coord in self.coords:
            param = nn.Parameter(torch.randn(1))
            self.coords.append(coord)
            self.params.append(param)
            optim.add_param_group({'params': param})
            print(optim)


if __name__ == '__main__':
    model = SpVarModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(1000):
        optim.zero_grad()

        coord_i = torch.Tensor([i % 3]).int()
        gt_i = torch.Tensor([coord_i / 2.0])

        model.add_param(coord_i, optim)

        out = model(coord_i)
        loss = torch.abs(gt_i - out)
        loss.backward()

        optim.step()

    for param in model.params:
        print(param)


