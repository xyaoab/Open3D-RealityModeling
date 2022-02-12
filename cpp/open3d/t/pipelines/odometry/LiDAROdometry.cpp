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
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Timer.h"
#include "open3d/visualization/utility/DrawGeometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

LiDARIntrinsicPtrs::LiDARIntrinsicPtrs(const LiDARIntrinsic& intrinsic) {
    // Specify config sent to kernels
    dir_lut_ptr =
            const_cast<float*>(intrinsic.unproj_dir_lut_.GetDataPtr<float>());
    offset_lut_ptr = const_cast<float*>(
            intrinsic.unproj_offset_lut_.GetDataPtr<float>());
    azimuth_lut_ptr =
            const_cast<float*>(intrinsic.azimuth_lut_.GetDataPtr<float>());
    altitude_lut_ptr =
            const_cast<float*>(intrinsic.altitude_lut_.GetDataPtr<float>());
    inv_altitude_lut_ptr = const_cast<int64_t*>(
            intrinsic.inv_altitude_lut_.GetDataPtr<int64_t>());

    width = intrinsic.width_;
    height = intrinsic.azimuth_lut_.GetLength();
    azimuth_resolution = (2 * M_PI) / width;
    inv_altitude_lut_length = intrinsic.inv_altitude_lut_.GetLength();
    inv_altitude_lut_resolution = intrinsic.inv_lut_resolution_;
}

LiDARIntrinsic::LiDARIntrinsic(const std::string& config_npz_file,
                               const core::Device& device) {
    auto result = t::io::ReadNpz(config_npz_file);

    // TODO: (store config in the calib file)
    int height = 128;
    int width = 1024;
    int n = 27.67;

    // Transformations always live on CPU
    lidar_to_sensor_ = result.at("lidar_to_sensor").To(core::Dtype::Float64);
    azimuth_lut_ = result.at("azimuth_table").To(device, core::Dtype::Float32);
    altitude_lut_ =
            result.at("altitude_table").To(device, core::Dtype::Float32);

    auto key0 = core::TensorKey::Slice(0, 3, 1);
    auto key1 = core::TensorKey::Slice(3, 4, 1);
    core::Tensor tt = lidar_to_sensor_.GetItem({key0, key1}) / range_scale_;
    lidar_to_sensor_.SetItem({key0, key1}, tt);

    sensor_to_lidar_ = lidar_to_sensor_.Inverse();

    // Basic rigid transform
    using core::Dtype;
    using core::None;
    using core::TensorKey;

    auto R = lidar_to_sensor_.GetItem(
            {TensorKey::Slice(0, 3, 1), TensorKey::Slice(0, 3, 1)});
    auto t = lidar_to_sensor_.GetItem(
            {TensorKey::Slice(0, 3, 1), TensorKey::Slice(3, 4, 1)});

    // Load azimuth and altitude LUT.

    // (1, w)
    auto theta_encoder = core::Tensor::Arange(2 * M_PI, 0.0, -2 * M_PI / width,
                                              Dtype::Float32, device)
                                 .View({1, -1});

    // (h, 1); deg2rad conversion
    auto theta_azimuth = (-(M_PI / 180.0) * azimuth_lut_).View({-1, 1});
    auto phi = ((M_PI / 180.0) * altitude_lut_).View({-1, 1});

    // Broadcast and convert to spherica
    auto theta = theta_encoder + theta_azimuth;
    auto x_dir = theta.Cos() * phi.Cos();
    auto y_dir = theta.Sin() * phi.Cos();
    auto z_dir = phi.Sin();

    auto dir_lut = core::Tensor({height, width, 3}, Dtype::Float32, device);

    std::vector<TensorKey> tks = {TensorKey::Slice(None, None, None),
                                  TensorKey::Slice(None, None, None),
                                  TensorKey::Index(0)};
    tks[2] = TensorKey::Index(0);
    dir_lut.SetItem(tks, x_dir);
    tks[2] = TensorKey::Index(1);
    dir_lut.SetItem(tks, y_dir);
    tks[2] = TensorKey::Index(2);
    dir_lut.SetItem(tks, z_dir);
    dir_lut /= range_scale_;
    dir_lut = (dir_lut.View({-1, 3}).Matmul(R.To(device, core::Float32).T()))
                      .View({height, width, 3})
                      .Contiguous();

    auto x_offset = n * (theta_encoder.Cos() - x_dir);
    auto y_offset = n * (theta_encoder.Sin() - y_dir);
    auto z_offset = (-n) * z_dir;
    auto offset_lut = core::Tensor({height, width, 3}, Dtype::Float32, device);
    tks[2] = TensorKey::Index(0);
    offset_lut.SetItem(tks, x_offset);
    tks[2] = TensorKey::Index(1);
    offset_lut.SetItem(tks, y_offset);
    tks[2] = TensorKey::Index(2);
    offset_lut.SetItem(tks, z_offset);
    offset_lut /= range_scale_;
    offset_lut += t.To(device, core::Float32).T();

    unproj_dir_lut_ = dir_lut.Clone();
    unproj_offset_lut_ = offset_lut.Clone();

    // First map [0, inv_lut_size] to [min - padding, max + padding]
    // Then linear search the nearest neighbor in the altitude table
    auto reversed_altitude_lut = altitude_lut_.Reverse();
    float max_altitude = reversed_altitude_lut[-1].Item<float>();
    float min_altitude = reversed_altitude_lut[0].Item<float>();
    int inv_table_size =
            int(std::ceil(max_altitude - min_altitude) / inv_lut_resolution_);
    std::vector<int64_t> inv_lut_data(inv_table_size);

    int i = 0;
    for (int j = 0; j < height - 1; ++j) {
        float thr = (reversed_altitude_lut[j].Item<float>() +
                     reversed_altitude_lut[j + 1].Item<float>()) *
                    0.5f;
        while (i * inv_lut_resolution_ + min_altitude < thr) {
            inv_lut_data[i] = j;
            ++i;
        }
    }
    for (; i < inv_table_size; ++i) {
        inv_lut_data[i] = height - 1;
    }

    core::Tensor inv_lut_table(inv_lut_data, {inv_table_size}, core::Int64,
                               device);
    inv_altitude_lut_ = inv_lut_table.Clone();
}

/// Return xyz-image and mask_image
/// Input: range image in UInt16
std::tuple<core::Tensor, core::Tensor> LiDARIntrinsic::Unproject(
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

    kernel::odometry::LiDARUnproject(range_image, transformation,
                                     LiDARIntrinsicPtrs(*this), xyz_im, mask_im,
                                     range_scale_, depth_min, depth_max);

    return std::make_tuple(xyz_im, mask_im);
}

/// Return u, v, r, mask
std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor>
LiDARIntrinsic::Project(const core::Tensor& xyz,
                        const core::Tensor& transformation) const {
    core::Device device = xyz.GetDevice();

    int64_t n = xyz.GetLength();
    core::Tensor u(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor v(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor r(core::SizeVector{n}, core::Dtype::Float32, device);
    core::Tensor mask(core::SizeVector{n}, core::Dtype::Bool, device);

    // sensor_to_lidar @ transformation @ xyz
    core::Tensor trans = sensor_to_lidar_.Matmul(transformation);
    kernel::odometry::LiDARProject(xyz, trans, LiDARIntrinsicPtrs(*this), u, v,
                                   r, mask);

    return std::make_tuple(u, v, r, mask);
}

// for i in range(h):
//   s = np.round(self.azimuth_table[i * factor] * self.w /
//                360.0).astype(int)
//   if s < 0:
//          s += self.w

//          shifted[i, s:] = im[i, :(w - s)]
//          shifted[i, :s] = im[i, (w - s):]
core::Tensor LiDARIntrinsic::Visualize(const core::Tensor& range) const {
    using core::None;
    using core::TensorKey;
    auto ans = core::Tensor::Empty(range.GetShape(), range.GetDtype(),
                                   range.GetDevice());
    auto shape = range.GetShape();
    int h = shape[0];
    int w = shape[1];

    for (int i = 0; i < h; ++i) {
        int s = std::round(azimuth_lut_[i].Item<float>() * w / 360.0);
        s += (s < 0) * w;

        auto rhs = range.GetItem(
                {TensorKey::Index(i), TensorKey::Slice(None, w - s, None)});
        ans.SetItem({TensorKey::Index(i), TensorKey::Slice(s, None, None)},
                    rhs);

        rhs = range.GetItem(
                {TensorKey::Index(i), TensorKey::Slice(w - s, None, None)});
        ans.SetItem({TensorKey::Index(i), TensorKey::Slice(None, s, None)},
                    rhs);
    }
    return ans;
}

core::Tensor GetNormalMap(const t::geometry::Image& im,
                          const LiDARIntrinsic& calib) {
    core::Tensor vertex_map, mask_map;
    std::tie(vertex_map, mask_map) = calib.Unproject(im.AsTensor());

    t::geometry::PointCloud pcd(vertex_map.IndexGet({mask_map}));

    core::Tensor normal_map =
            core::Tensor::Zeros(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());

    pcd.EstimateNormals();
    normal_map.IndexSet({mask_map}, pcd.GetPointNormals());
    return normal_map;
}

OdometryResult LiDAROdometry(const Image& source,
                             const Image& target,
                             const LiDARIntrinsic& calib,
                             const core::Tensor& init_source_to_target,
                             const float depth_min,
                             const float depth_max,
                             const float dist_diff,
                             const OdometryConvergenceCriteria& criteria) {
    core::Tensor target_normal_map = GetNormalMap(target, calib);
    return LiDAROdometry(source, target, target_normal_map, calib,
                         init_source_to_target, depth_min, depth_max, dist_diff,
                         criteria);
}

OdometryResult LiDAROdometry(const Image& source,
                             const Image& target,
                             const core::Tensor& target_normal_map,
                             const LiDARIntrinsic& calib,
                             const core::Tensor& init_source_to_target,
                             const float depth_min,
                             const float depth_max,
                             const float dist_diff,
                             const OdometryConvergenceCriteria& criteria) {
    core::Tensor source_vertex_map, source_mask_map;
    core::Tensor target_vertex_map, target_mask_map;

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());
    std::tie(source_vertex_map, source_mask_map) =
            calib.Unproject(source.AsTensor(), identity, depth_min, depth_max);
    std::tie(target_vertex_map, target_mask_map) =
            calib.Unproject(target.AsTensor(), identity, depth_min, depth_max);

    OdometryResult result(init_source_to_target);
    for (int i = 0; i < criteria.max_iteration_; ++i) {
        OdometryResult delta_result = ComputeLiDAROdometryPointToPlane(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, calib,
                result.transformation_, dist_diff);
        result.transformation_ =
                (delta_result.transformation_.Matmul(result.transformation_))
                        .Contiguous();
        utility::LogDebug("iter {}: rmse = {}, fitness = {}", i,
                          delta_result.inlier_rmse_, delta_result.fitness_);
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
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_diff) {
    core::Tensor se3_delta;
    float inlier_residual;
    int inlier_count;

    kernel::odometry::ComputeLiDAROdometryPointToPlane(
            source_vertex_map, source_mask_map, target_vertex_map,
            target_mask_map, target_normal_map, init_source_to_target,
            calib.sensor_to_lidar_, LiDARIntrinsicPtrs(calib), se3_delta,
            inlier_residual, inlier_count, depth_diff);

    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }

    return OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
