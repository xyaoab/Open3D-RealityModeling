# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/integrate.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_testdata


def visualize_blocks(block_coords, block_len, color, offset_rate=0.01):
    # buf_indices = vbg.hashmap().active_buf_indices()
    xyz000 = (block_coords.cpu().numpy() + offset_rate)* block_len
    # (vbg.hashmap().key_tensor()[buf_indices]).to(
    # o3c.float32) * block_len
    xyz001 = xyz000 + np.array([[block_len, 0, 0]])
    xyz010 = xyz000 + np.array([[0, block_len, 0]])
    xyz100 = xyz000 + np.array([[0, 0, block_len]])
    xyz = np.concatenate((xyz000, xyz001, xyz010, xyz100),
                         axis=0).astype(np.float32)

    lineset = o3d.t.geometry.LineSet()
    lineset.point['positions'] = o3c.Tensor(xyz)

    n = len(xyz000)
    lineset000 = np.arange(0, n)
    lineset001 = np.arange(n, 2 * n)
    lineset010 = np.arange(2 * n, 3 * n)
    lineset100 = np.arange(3 * n, 4 * n)

    indices001 = np.stack((lineset000, lineset001), axis=1)
    indices010 = np.stack((lineset000, lineset010), axis=1)
    indices100 = np.stack((lineset000, lineset100), axis=1)
    indices = np.concatenate((indices001, indices010, indices100), axis=0)

    lineset.line['indices'] = o3c.Tensor(indices.astype(np.int32))
    colors = np.tile(color, (3 * n, 1))
    lineset.line['colors'] = o3c.Tensor(colors.astype(np.float32))
    return lineset


def integrate(depth_file_names, color_file_names, depth_intrinsic,
              color_intrinsic, extrinsics, config):
    if os.path.exists(config.path_npz):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.path_npz))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
        print(vbg.hashmap().size())
        buf_indices = vbg.hashmap().active_buf_indices()
        global_block_coords = vbg.hashmap().key_tensor()[buf_indices].to(o3c.float32)
        for i in range(3):
            print(global_block_coords[:, i].min().item(), global_block_coords[:, i].max().item())

        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.path_npz))

        n_files = len(depth_file_names)
        device = o3d.core.Device(config.device)

        if config.integrate_color:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=0.1,
                block_resolution=16,
                block_count=50000,
                device=device)
        else:
            vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
                                                attr_dtypes=(o3c.float32,
                                                             o3c.float32),
                                                attr_channels=((1), (1)),
                                                voxel_size=0.1,
                                                block_resolution=16,
                                                block_count=50000,
                                                device=device)

        start = time.time()
        for i in range(n_files):
            print('Integrating frame {}/{}'.format(i, n_files))

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, depth_intrinsic, extrinsic, config.depth_scale,
                config.depth_max)

            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                vbg.integrate(frustum_block_coords, depth, color,
                              depth_intrinsic, color_intrinsic, extrinsic,
                              config.depth_scale, config.depth_max)
            else:
                vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
                              extrinsic, config.depth_scale, config.depth_max)

            if i % 5000 == 0:
                lineset = o3d.geometry.LineSet.create_camera_visualization(
                    640,
                    480,
                    depth_intrinsic.cpu().numpy(),
                    extrinsic.cpu().numpy(),
                    scale=0.5)

                block_len = 16 * 3.0 / 512
                local_blocks = visualize_blocks(frustum_block_coords, block_len,
                                                np.array([1, 0, 0]), offset_rate=0)

                buf_indices = vbg.hashmap().active_buf_indices()
                global_block_coords = vbg.hashmap().key_tensor(
                )[buf_indices].to(o3c.float32)
                global_blocks = visualize_blocks(global_block_coords, block_len,
                                                 np.array([0, 1, 0]))
                pcd = vbg.extract_point_cloud(3)

                o3d.visualization.draw(
                    [pcd, global_blocks, local_blocks, lineset])

            dt = time.time() - start
        print('Finished integrating {} frames in {} seconds'.format(
            n_files, dt))
        print('Saving to {}...'.format(config.path_npz))
        vbg.save(config.path_npz)
        print('Saving finished')

    return vbg


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--integrate_color', action='store_true')
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

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    depth_intrinsic = load_intrinsic(config)
    color_intrinsic = load_intrinsic(config, 'color')

    extrinsics = load_extrinsics(config.path_trajectory, config)
    vbg = integrate(depth_file_names, color_file_names, depth_intrinsic,
                    color_intrinsic, extrinsics, config)

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])
