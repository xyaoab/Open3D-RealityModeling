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

/// \file RGBDOdometry.h
/// All the 4x4 transformation in this file, from params to returns, are
/// Float64. Only convert to Float32 in kernel calls.

#include "open3d/t/pipelines/odometry/LiDAROdometry.h"

#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/t/pipelines/kernel/LiDAROdometry.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

LiDARCalib::LiDARCalib(const std::string& config_npz_file,
                       const core::Device& device) {
    auto result = t::io::ReadNpz(config_npz_file);

    // Transformations always live on CPU
    lidar_to_sensor_ = result.at("lidar_to_sensor").To(core::Dtype::Float64);
    sensor_to_lidar_ = lidar_to_sensor_.Inverse();
    auto key0 = core::TensorKey::Slice(0, 3, 1);
    auto key1 = core::TensorKey::Slice(3, 4, 1);
    core::Tensor t = sensor_to_lidar_.GetItem({key0, key1}) / range_scale_;
    sensor_to_lidar_.SetItem({key0, key1}, t);

    azimuth_lut_ = result.at("azimuth_table").To(device, core::Dtype::Float32);
    altitude_lut_ =
            result.at("altitude_table").To(device, core::Dtype::Float32);
    inv_altitude_lut_ =
            result.at("inv_altitude_table").To(device, core::Dtype::Int64);

    unproj_dir_lut_ = result.at("lut_dir").To(device, core::Dtype::Float32);
    unproj_offset_lut_ =
            result.at("lut_offset").To(device, core::Dtype::Float32);
}

/// Return xyz-image and mask_image
/// Input: range image in UInt16
std::tuple<core::Tensor, core::Tensor> LiDARCalib::Unproject(
        const core::Tensor& range_image,
        const core::Tensor& transformation,
        float depth_min,
        float depth_max) const {
    // TODO: shape check
    auto sv = range_image.GetShape();
    int64_t h = sv[0];
    int64_t w = sv[1];

    core::Device device = range_image.GetDevice();
    core::Tensor xyz_im(core::SizeVector{h, w, 3}, core::Dtype::Float32,
                        device);
    core::Tensor mask_im(core::SizeVector{h, w}, core::Dtype::Bool, device);

    kernel::odometry::LiDARUnproject(
            range_image, transformation, unproj_dir_lut_, unproj_offset_lut_,
            xyz_im, mask_im, range_scale_, depth_min, depth_max);

    return std::make_tuple(xyz_im, mask_im);
}

/// Return u, v, r, mask
std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor>
LiDARCalib::Project(const core::Tensor& xyz,
                    const core::Tensor& transformation) const {
    core::Device device = xyz.GetDevice();

    int64_t n = xyz.GetLength();
    core::Tensor u(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor v(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor r(core::SizeVector{n}, core::Dtype::Float32, device);
    core::Tensor mask(core::SizeVector{n}, core::Dtype::Bool, device);

    // Left-multiply two 4x4 matrices.
    core::Tensor trans = sensor_to_lidar_.Matmul(transformation);
    kernel::odometry::LiDARProject(xyz, trans, azimuth_lut_, altitude_lut_,
                                   inv_altitude_lut_, u, v, r, mask);

    return std::make_tuple(u, v, r, mask);
}

OdometryResult LiDAROdometry(const Image& source,
                             const Image& target,
                             const LiDARCalib& calib,
                             const core::Tensor& init_source_to_target,
                             const float depth_min,
                             const float depth_max,
                             const float dist_diff,
                             const OdometryConvergenceCriteria& criteria) {
    core::Device device = source.GetDevice();
    core::Tensor source_vertex_map, source_mask_map;
    core::Tensor target_vertex_map, target_mask_map;

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());
    std::tie(source_vertex_map, source_mask_map) =
            calib.Unproject(source.AsTensor(), identity, depth_min, depth_max);
    std::tie(target_vertex_map, target_mask_map) =
            calib.Unproject(target.AsTensor(), identity, depth_min, depth_max);

    /// TODO: normal map estimate from kernels
    t::geometry::PointCloud target_pcd(
            target_vertex_map.Reshape(core::SizeVector{-1, 3}));
    target_pcd = target_pcd.To(core::Device());
    target_pcd.EstimateNormals();
    visualization::DrawGeometries(
            {std::make_shared<open3d::geometry::PointCloud>(
                    target_pcd.ToLegacy())});

    core::Tensor target_normal_map =
            target_pcd.GetPointNormals()
                    .Reshape(target_vertex_map.GetShape())
                    .To(device);

    OdometryResult result(init_source_to_target);
    for (int i = 0; i < criteria.max_iteration_; ++i) {
        OdometryResult delta_result = ComputeLiDAROdometryPointToPlane(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, calib,
                result.transformation_, dist_diff);
        result.transformation_ =
                delta_result.transformation_.Matmul(result.transformation_);
    }
    return result;
}

OdometryResult ComputeLiDAROdometryPointToPlane(
        const core::Tensor& source_vertex_map,
        const core::Tensor& source_mask_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        // Note: currently target_normal_map is from point cloud
        const core::Tensor& target_normal_map,
        const LiDARCalib& calib,
        const core::Tensor& init_source_to_target,
        const float depth_diff) {
    return OdometryResult();
    // return kernel::odometry::ComputeLiDAROdometryPointToPlane(
    //         source_vertex_map, source_mask_map, target_vertex_map,
    //         target_mask_map, target_normal_map, init_source_to_target,
    //         depth_diff);
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
