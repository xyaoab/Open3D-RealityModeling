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
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/io/NumpyIO.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

LiDARCalib::LiDARCalib(const std::string& config_npz_file) {
    auto result = t::io::ReadNpz(config_npz_file);

    lidar_to_sensor_ = result.at("lidar_to_sensor");

    azimuth_lut_ = result.at("azimuth_table");
    altitude_lut_ = result.at("altitude_table");
    inv_altitude_lut_ = result.at("inv_altitude_table");

    unproj_dir_lut_ = result.at("lut_dir");
    unproj_dir_lut_ = result.at("lut_offset");
}

/// Return xyz-image and mask_image
/// Input: range image in UInt16
std::tuple<core::Tensor, core::Tensor> LiDARCalib::Unproject(
        core::Tensor& range_image, float scale_factor) {
    return std::make_tuple(core::Tensor(), core::Tensor());
}

/// Return u, v, r, mask
std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor>
LiDARCalib::Project(const core::Tensor& xyz,
                    const core::Tensor& transformation) {
    return std::make_tuple(core::Tensor(), core::Tensor(), core::Tensor(),
                           core::Tensor());
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
