// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/VoxelBlockGrid.h"

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace tests {

using namespace t::geometry;

class VoxelBlockGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(VoxelBlockGrid,
                         VoxelBlockGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

static core::Tensor GetIntrinsicTensor() {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    return core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});
}

static std::vector<core::Tensor> GetExtrinsicTensors() {
    // Extrinsics
    std::string trajectory_path =
            std::string(TEST_DATA_DIR) + "/RGBD/odometry.log";
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    std::vector<core::Tensor> extrinsics;
    for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
        Eigen::Matrix4d extrinsic = trajectory->parameters_[i].extrinsic_;
        core::Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic);
        extrinsics.emplace_back(extrinsic_t);
    }

    return extrinsics;
}

TEST_P(VoxelBlockGridPermuteDevices, Construct) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    for (auto backend : backends) {
        auto vbg = VoxelBlockGrid({"tsdf", "weight", "color"},
                                  {core::Float32, core::UInt16, core::UInt8},
                                  {{1}, {1}, {3}}, 3.0 / 512, 8, 10, device,
                                  backend);

        auto hashmap = vbg.GetHashMap();

        auto tsdf_tensor = hashmap.GetValueTensor(0);
        auto weight_tensor = hashmap.GetValueTensor(1);
        auto color_tensor = hashmap.GetValueTensor(2);

        utility::LogInfo("TSDF block shape = {}, dtype = {}",
                         tsdf_tensor.GetShape(),
                         tsdf_tensor.GetDtype().ToString());
        utility::LogInfo("Weight block shape = {}, dtype = {}",
                         weight_tensor.GetShape(),
                         weight_tensor.GetDtype().ToString());
        utility::LogInfo("Color block shape = {}, dtype = {}",
                         color_tensor.GetShape(),
                         color_tensor.GetDtype().ToString());
    }
}

TEST_P(VoxelBlockGridPermuteDevices, Indexing) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    for (auto backend : backends) {
        auto vbg = VoxelBlockGrid({"tsdf", "weight", "color"},
                                  {core::Float32, core::UInt16, core::UInt8},
                                  {{1}, {1}, {3}}, 3.0 / 512, 2, 10, device,
                                  backend);

        auto hashmap = vbg.GetHashMap();
        core::Tensor keys = core::Tensor(std::vector<int>{-1, 3, 2, 0, 2, 4},
                                         core::SizeVector{2, 3},
                                         core::Dtype::Int32, device);

        core::Tensor buf_indices, masks;
        hashmap.Activate(keys, buf_indices, masks);
        utility::LogInfo("Block indices: {}",
                         buf_indices.IndexGet({masks}).ToString());

        core::Tensor voxel_indices = vbg.GetVoxelIndices();
        utility::LogInfo("Voxel indices for advanced indexing: {}",
                         voxel_indices.ToString());

        core::Tensor voxel_coords = vbg.GetVoxelCoordinates(voxel_indices);
        utility::LogInfo("Voxel coordinates for positioning: {}",
                         voxel_coords.ToString());
    }
}

TEST_P(VoxelBlockGridPermuteDevices, GetUniqueBlockCoordinates) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::StdGPU);
        backends.push_back(core::HashBackendType::Slab);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    core::Tensor intrinsic = GetIntrinsicTensor();
    std::vector<core::Tensor> extrinsics = GetExtrinsicTensors();
    const float depth_scale = 1000.0;
    const float depth_max = 3.0;

    for (auto backend : backends) {
        auto vbg = VoxelBlockGrid({"tsdf", "weight", "color"},
                                  {core::Float32, core::Float32, core::UInt16},
                                  {{1}, {1}, {3}}, 3.0 / 512, 8, 10000, device,
                                  backend);

        const int i = 0;
        Image depth = t::io::CreateImageFromFile(
                              fmt::format("{}/RGBD/depth/{:05d}.png",
                                          std::string(TEST_DATA_DIR), i))
                              ->To(device);
        core::Tensor block_coords_from_depth = vbg.GetUniqueBlockCoordinates(
                depth, intrinsic, extrinsics[i], depth_scale, depth_max);

        PointCloud pcd = PointCloud::CreateFromDepthImage(
                depth, intrinsic, extrinsics[i], depth_scale, depth_max, 4);
        core::Tensor block_coords_from_pcd = vbg.GetUniqueBlockCoordinates(pcd);

        utility::LogInfo("Number of blocks from depth: {}, blocks from pcd: {}",
                         block_coords_from_depth.GetLength(),
                         block_coords_from_pcd.GetLength());
    }
}

TEST_P(VoxelBlockGridPermuteDevices, Integrate) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    core::Tensor intrinsic = GetIntrinsicTensor();
    std::vector<core::Tensor> extrinsics = GetExtrinsicTensors();
    const float depth_scale = 1000.0;
    const float depth_min = 0.1;
    const float depth_max = 3.0;
    for (auto backend : backends) {
        for (int block_resolution : std::vector<int>{8, 16}) {
            for (auto &dtype :
                 std::vector<core::Dtype>{core::Float32, core::UInt16}) {
                auto vbg = VoxelBlockGrid(
                        {"tsdf", "weight", "color"},
                        {core::Float32, dtype, dtype}, {{1}, {1}, {3}},
                        3.0 / 512, block_resolution, 10000, device, backend);

                for (size_t i = 0; i < extrinsics.size(); ++i) {
                    Image depth =
                            t::io::CreateImageFromFile(
                                    fmt::format("{}/RGBD/depth/{:05d}.png",
                                                std::string(TEST_DATA_DIR), i))
                                    ->To(device);
                    Image color =
                            t::io::CreateImageFromFile(
                                    fmt::format("{}/RGBD/color/{:05d}.jpg",
                                                std::string(TEST_DATA_DIR), i))
                                    ->To(device);

                    core::Tensor frustum_block_coords =
                            vbg.GetUniqueBlockCoordinates(
                                    depth, intrinsic, extrinsics[i],
                                    depth_scale, depth_max);
                    vbg.Integrate(frustum_block_coords, depth, color, intrinsic,
                                  extrinsics[i]);

                    // Check raycasting with non-slab backends.
                    if (i == extrinsics.size() - 1 &&
                        backend != core::HashBackendType::Slab) {
                        auto result = vbg.RayCast(
                                frustum_block_coords, intrinsic, extrinsics[i],
                                depth.GetCols(), depth.GetRows(), depth_scale,
                                depth_min, depth_max, 1.0);

                        t::geometry::Image normal_raycast(result["normal"]);
                        visualization::DrawGeometries(
                                {std::make_shared<open3d::geometry::Image>(
                                        normal_raycast.ToLegacy())});
                        normal_raycast.AsTensor().Save("normal.npy");

                        core::Tensor range_map = result["range"];
                        t::geometry::Image im_near(
                                range_map.Slice(2, 0, 1).Contiguous() /
                                depth_max);
                        visualization::DrawGeometries(
                                {std::make_shared<open3d::geometry::Image>(
                                        im_near.ToLegacy())});

                        t::geometry::Image depth_raycast(result["depth"]);
                        visualization::DrawGeometries(
                                {std::make_shared<open3d::geometry::Image>(
                                        depth_raycast
                                                .ColorizeDepth(depth_scale,
                                                               depth_min,
                                                               depth_max)
                                                .ToLegacy())});
                        t::geometry::Image color_raycast(result["color"]);
                        visualization::DrawGeometries(
                                {std::make_shared<open3d::geometry::Image>(
                                        color_raycast.ToLegacy())});

                        core::Tensor index_map = result["index"];
                        core::Tensor mask_map = result["mask"];
                        core::Tensor ratio_map = result["ratio"];
                        core::Tensor tsdf = vbg.GetAttribute("tsdf");
                        core::Tensor weight = vbg.GetAttribute("weight");
                        core::Tensor color = vbg.GetAttribute("color");
                        t::io::WriteNpz(
                                "raycasting.npz",
                                std::unordered_map<std::string, core::Tensor>{
                                        {"index", index_map},
                                        {"mask", mask_map},
                                        {"ratio", ratio_map},
                                        {"tsdf", tsdf},
                                        {"weight", weight},
                                        {"color", color}});

                        // Sanity check in python:
                    }
                }

                // Check customized indexing and processing -- kernelized
                {
                    core::Tensor voxel_coords, voxel_indices;
                    std::tie(voxel_coords, voxel_indices) =
                            vbg.GetVoxelCoordinatesAndFlattenedIndices();

                    core::Tensor tsdf =
                            vbg.GetAttribute("tsdf").View({-1}).IndexGet(
                                    {voxel_indices});
                    core::Tensor weight =
                            vbg.GetAttribute("weight").View({-1}).IndexGet(
                                    {voxel_indices});

                    core::Tensor mask = tsdf.Abs().Lt(0.3) && weight.Gt(1);
                    core::Tensor valid_coords = voxel_coords.IndexGet({mask});

                    PointCloud valid_pcd(valid_coords);
                    auto vis_valid_pcd =
                            std::make_shared<open3d::geometry::PointCloud>(
                                    valid_pcd.ToLegacy());
                    visualization::DrawGeometries({vis_valid_pcd});
                }

                // Check customized indexing and processing -- non-kernelized
                {
                    core::Tensor voxel_indices = vbg.GetVoxelIndices();
                    core::Tensor voxel_coords =
                            vbg.GetVoxelCoordinates(voxel_indices);
                    core::Tensor tsdf = vbg.GetAttribute("tsdf")
                                                .IndexGet({voxel_indices[0],
                                                           voxel_indices[3],
                                                           voxel_indices[2],
                                                           voxel_indices[1]})
                                                .View({-1});
                    core::Tensor weight = vbg.GetAttribute("weight")
                                                  .IndexGet({voxel_indices[0],
                                                             voxel_indices[3],
                                                             voxel_indices[2],
                                                             voxel_indices[1]})
                                                  .View({-1});

                    core::Tensor mask = tsdf.Abs().Lt(0.3) && weight.Gt(1);
                    core::Tensor valid_coords =
                            voxel_coords.T().IndexGet({mask});
                    PointCloud valid_pcd(valid_coords);
                    auto vis_valid_pcd =
                            std::make_shared<open3d::geometry::PointCloud>(
                                    valid_pcd.ToLegacy());
                    visualization::DrawGeometries({vis_valid_pcd});
                }

                // Check surface extraction
                auto pcd = std::make_shared<open3d::geometry::PointCloud>(
                        vbg.ExtractPointCloud().ToLegacy());
                visualization::DrawGeometries({pcd});

                auto mesh = std::make_shared<open3d::geometry::TriangleMesh>(
                        vbg.ExtractTriangleMesh(-1, 0.0).ToLegacy());
                visualization::DrawGeometries({mesh});

                vbg.Save("tmp.npz");
                auto vbg_new = VoxelBlockGrid::Load("tmp.npz");
                auto pcd_new = std::make_shared<open3d::geometry::PointCloud>(
                        vbg_new.ExtractPointCloud().ToLegacy());
                visualization::DrawGeometries({pcd_new});
            }
        }
    }
}
}  // namespace tests
}  // namespace open3d
