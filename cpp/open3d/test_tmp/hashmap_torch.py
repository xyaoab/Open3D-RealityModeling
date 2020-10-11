import torch
import torch.nn as nn
from torch.utils import dlpack
import open3d as o3d

if __name__ == '__main__':
    # Construct hashmap
    capacity = 10
    hashmap = o3d.core.Hashmap(init_capacity=10,
                               dtype_key=o3d.core.Dtype.Int32,
                               dtype_value=o3d.core.Dtype.Float32,
                               device=o3d.core.Device('cuda:0'))

    # Get keys
    key_blob = hashmap.get_key_blob_as_tensor((10,), o3d.core.Dtype.Int32)
    value_blob = hashmap.get_value_blob_as_tensor((10,), o3d.core.Dtype.Float32)

    # Wrap them with a torch parameter
    key_tensor = dlpack.from_dlpack(key_blob.to_dlpack())
    value_tensor = nn.Parameter(dlpack.from_dlpack(value_blob.to_dlpack()))
    optimizer = torch.optim.SGD([value_tensor], lr=0.01)

    # Insert in hashmap and check synchronization
    coords = o3d.core.Tensor(
        [[100], [200], [300], [200], [400], [100], [500], [200]],
        dtype=o3d.core.Dtype.Int32).cuda()
    values = o3d.core.Tensor(
        [[1.0], [2.0], [3.0], [2.0], [4.0], [1.0], [5.0], [2.0]],
        dtype=o3d.core.Dtype.Float32).cuda()
    iterators, masks = hashmap.insert(coords, values)
    masks_tensor = dlpack.from_dlpack(
        masks.to(o3d.core.Dtype.UInt8).to_dlpack()).bool()

    # Apply some naive differentiable computation here
    optimizer.zero_grad()
    loss = value_tensor.abs().sum()
    loss.backward()
    optimizer.step()

    print(value_tensor)
    print(value_blob)

