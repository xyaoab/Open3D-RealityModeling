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

#include "open3d/t/geometry/LiDARImage.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/t/pipelines/kernel/LiDAROdometry.h"
namespace open3d {
namespace t {
namespace geometry {

LiDARIntrinsic::LiDARIntrinsic(int width,
                               int height,
                               float min_altitude,
                               float max_altitude,
                               const core::Tensor& lidar_to_sensor)
    : width_(width),
      height_(height),
      min_altitude_(min_altitude),
      max_altitude_(max_altitude),
      has_lut_(false) {
    lidar_to_sensor_ = lidar_to_sensor.Clone();
    auto key0 = core::TensorKey::Slice(0, 3, 1);
    auto key1 = core::TensorKey::Slice(3, 4, 1);
    core::Tensor tt = lidar_to_sensor_.GetItem({key0, key1}) / range_scale_;
    lidar_to_sensor_.SetItem({key0, key1}, tt);

    sensor_to_lidar_ = lidar_to_sensor_.Inverse();
}

LiDARIntrinsic::LiDARIntrinsic(const std::string& config_npz_file,
                               const core::Device& device) {
    auto result = t::io::ReadNpz(config_npz_file);

    has_lut_ = true;

    height_ = result.at("height").To(core::Int32).Item<int>();
    width_ = result.at("width").To(core::Int32).Item<int>();
    float n = result.at("n").To(core::Float32).Item<float>();

    // Transformations always live on CPU
    azimuth_lut_ = result.at("azimuth_table").To(device, core::Dtype::Float32);
    altitude_lut_ =
            result.at("altitude_table").To(device, core::Dtype::Float32);
    min_altitude_ = altitude_lut_[-1].Item<float>();
    max_altitude_ = altitude_lut_[0].Item<float>();

    lidar_to_sensor_ = result.at("lidar_to_sensor").To(core::Dtype::Float64);
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
    auto theta_encoder = core::Tensor::Arange(2 * M_PI, 0.0, -2 * M_PI / width_,
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

    auto dir_lut = core::Tensor({height_, width_, 3}, Dtype::Float32, device);

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
                      .View({height_, width_, 3})
                      .Contiguous();

    auto x_offset = n * (theta_encoder.Cos() - x_dir);
    auto y_offset = n * (theta_encoder.Sin() - y_dir);
    auto z_offset = (-n) * z_dir;
    auto offset_lut =
            core::Tensor({height_, width_, 3}, Dtype::Float32, device);
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
    int inv_table_size =
            int(std::ceil(max_altitude_ - min_altitude_) / inv_lut_resolution_);
    std::vector<int64_t> inv_lut_data(inv_table_size);

    int i = 0;
    for (int j = 0; j < height_ - 1; ++j) {
        float thr = (reversed_altitude_lut[j].Item<float>() +
                     reversed_altitude_lut[j + 1].Item<float>()) *
                    0.5f;
        while (i * inv_lut_resolution_ + min_altitude_ < thr) {
            inv_lut_data[i] = j;
            ++i;
        }
    }
    for (; i < inv_table_size; ++i) {
        inv_lut_data[i] = height_ - 1;
    }

    core::Tensor inv_lut_table(inv_lut_data, {inv_table_size}, core::Int64,
                               device);
    inv_altitude_lut_ = inv_lut_table.Clone();
}

LiDARIntrinsicPtrs::LiDARIntrinsicPtrs(const LiDARIntrinsic& intrinsic) {
    width = intrinsic.width_;
    height = intrinsic.height_;
    min_altitude = intrinsic.min_altitude_;
    max_altitude = intrinsic.max_altitude_;
    azimuth_resolution = (2 * M_PI) / width;

    has_lut = intrinsic.has_lut_;

    if (has_lut) {
        // Specify config sent to kernels
        dir_lut_ptr = const_cast<float*>(
                intrinsic.unproj_dir_lut_.GetDataPtr<float>());
        offset_lut_ptr = const_cast<float*>(
                intrinsic.unproj_offset_lut_.GetDataPtr<float>());
        azimuth_lut_ptr =
                const_cast<float*>(intrinsic.azimuth_lut_.GetDataPtr<float>());
        altitude_lut_ptr =
                const_cast<float*>(intrinsic.altitude_lut_.GetDataPtr<float>());
        inv_altitude_lut_ptr = const_cast<int64_t*>(
                intrinsic.inv_altitude_lut_.GetDataPtr<int64_t>());

        inv_altitude_lut_length = intrinsic.inv_altitude_lut_.GetLength();
        inv_altitude_lut_resolution = intrinsic.inv_lut_resolution_;
    }
}

/// Return xyz-image and mask_image
/// Input: range image in UInt16
std::tuple<core::Tensor, core::Tensor> LiDARImage::Unproject(
        const LiDARIntrinsic& intrinsic,
        const core::Tensor& transformation,
        float depth_min,
        float depth_max) const {
    // TODO: shape check
    auto sv = data_.GetShape();
    int64_t h = sv[0];
    int64_t w = sv[1];

    core::Device device = data_.GetDevice();
    core::Tensor xyz_im(core::SizeVector{h, w, 3}, core::Dtype::Float32,
                        device);
    core::Tensor mask_im(core::SizeVector{h, w}, core::Dtype::Bool, device);

    t::pipelines::kernel::odometry::LiDARUnproject(
            data_, transformation, LiDARIntrinsicPtrs(intrinsic), xyz_im,
            mask_im, intrinsic.range_scale_, depth_min, depth_max);

    return std::make_tuple(xyz_im, mask_im);
}

/// Return u, v, r, mask
std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor>
LiDARImage::Project(const core::Tensor& xyz,
                    const LiDARIntrinsic& intrinsic,
                    const core::Tensor& transformation) {
    core::Device device = xyz.GetDevice();

    int64_t n = xyz.GetLength();
    core::Tensor u(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor v(core::SizeVector{n}, core::Dtype::Int64, device);
    core::Tensor r(core::SizeVector{n}, core::Dtype::Float32, device);
    core::Tensor mask(core::SizeVector{n}, core::Dtype::Bool, device);

    // sensor_to_lidar @ transformation @ xyz
    core::Tensor trans = intrinsic.sensor_to_lidar_.Matmul(transformation);
    t::pipelines::kernel::odometry::LiDARProject(
            xyz, trans, LiDARIntrinsicPtrs(intrinsic), u, v, r, mask);

    return std::make_tuple(u, v, r, mask);
}

Image LiDARImage::Visualize(const LiDARIntrinsic& intrinsic) const {
    if (!intrinsic.has_lut_) {
        return data_;
    }

    using core::None;
    using core::TensorKey;
    auto ans = core::Tensor::Empty(data_.GetShape(), data_.GetDtype(),
                                   data_.GetDevice());
    auto shape = data_.GetShape();
    int h = shape[0];
    int w = shape[1];

    for (int i = 0; i < h; ++i) {
        int s = std::round(intrinsic.azimuth_lut_[i].Item<float>() * w / 360.0);
        s += (s < 0) * w;

        auto rhs = data_.GetItem(
                {TensorKey::Index(i), TensorKey::Slice(None, w - s, None)});
        ans.SetItem({TensorKey::Index(i), TensorKey::Slice(s, None, None)},
                    rhs);

        rhs = data_.GetItem(
                {TensorKey::Index(i), TensorKey::Slice(w - s, None, None)});
        ans.SetItem({TensorKey::Index(i), TensorKey::Slice(None, s, None)},
                    rhs);
    }
    return t::geometry::Image(ans);
}

core::Tensor LiDARImage::GetNormalMap(const LiDARIntrinsic& calib) const {
    core::Tensor vertex_map, mask_map;
    std::tie(vertex_map, mask_map) = Unproject(calib);

    return GetNormalMap(vertex_map, mask_map, calib);
}

core::Tensor LiDARImage::GetNormalMap(const core::Tensor& vertex_map,
                                      const core::Tensor& mask_map,
                                      const LiDARIntrinsic& calib) {
    t::geometry::PointCloud pcd(vertex_map.IndexGet({mask_map}));

    core::Tensor normal_map =
            core::Tensor::Zeros(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());

    pcd.EstimateNormals();
    normal_map.IndexSet({mask_map}, pcd.GetPointNormals());
    return normal_map;
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
