# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import open3d as o3d
import time
import torch
from torch.utils.dlpack import from_dlpack
import matplotlib.pyplot as plt

import imageio
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigParser
from common import load_rgbd_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_testdata


def to_torch(o3d_tensor):
    return from_dlpack(o3d_tensor.to_dlpack())


def optimize(vbg, depth_file_names, color_file_names, intrinsic, extrinsics,
             config):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    ### differentiable rendering
    # Weight for reference
    weight_volume_th = to_torch(vbg.attribute('weight')).flatten()

    # Optimize TSDF AND color jointly
    tsdf_volume_th = to_torch(vbg.attribute('tsdf')).flatten()
    color_volume_th = to_torch(vbg.attribute('color')).view((-1, 3))

    tsdf_volume_th.requires_grad_(True)
    color_volume_th.requires_grad_(True)

    optimizer = torch.optim.Adam([tsdf_volume_th, color_volume_th], lr=1e-4)

    #### Try overfitting with images
    for step in range(10000):
        optimizer.zero_grad()

        k = np.random.randint(0, n_files)

        # GT depth (to accelerate rendering),
        # TODO: query all the blocks for better rendering
        depth = o3d.t.io.read_image(depth_file_names[k]).to(device)
        extrinsic = extrinsics[k]

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, config.depth_scale, config.depth_max)

        # Rendering from volumes
        rendering = vbg.ray_cast(frustum_block_coords, intrinsic, extrinsic,
                                 depth.columns, depth.rows,
                                 ['normal', 'mask', 'interp_ratio', 'index'],
                                 config.depth_scale, config.depth_min,
                                 config.depth_max, 1)
        normal_map = rendering['normal']

        # In-place modification used here, need to think how to make them differentiable
        mask = to_torch(rendering['mask'].to(o3d.core.Dtype.UInt8)).bool()
        ratio = to_torch(rendering['interp_ratio'])
        index = to_torch(rendering['index'])

        color_rendering = torch.zeros(depth.rows, depth.columns, 3).cuda()

        # Trilinear interpolation of color (RGB) from grid points
        for i in range(8):
            mask_i = mask[:, :, i]
            ratio_i = ratio[:, :, i]
            index_i = index[:, :, i]
            rhs = torch.unsqueeze(ratio_i[mask_i],
                                  -1) * color_volume_th[index_i[mask_i]]

            color_rendering[mask_i] += rhs / 255.0

        # GT color
        color_gt = to_torch(
            o3d.t.io.read_image(
                color_file_names[k]).to(device).as_tensor()).to(float) / 255.0

        loss = torch.abs(color_rendering - color_gt.cuda()).mean()
        loss.backward()
        optimizer.step()
        print('step {}, loss {}'.format(step, loss.item()))

if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config.path_dataset = get_default_testdata()
        config.path_trajectory = os.path.join(config.path_dataset,
                                              'trajectory.log')

    depth_file_names, color_file_names = load_rgbd_file_names(config)
    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
    optimize(vbg, depth_file_names, color_file_names, intrinsic, extrinsics,
             config)
