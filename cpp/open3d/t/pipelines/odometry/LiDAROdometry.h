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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

class LiDARCalib {
public:
    LiDARCalib(const std::string& config_npz_file, const core::Device& device);

    /// Return xyz-image and mask_image
    /// Input: range image in UInt16
    std::tuple<core::Tensor, core::Tensor> Unproject(core::Tensor& range_image,
                                                     float depth_scale = 1000.0,
                                                     float depth_min = 0.65,
                                                     float detph_max = 10.0);

    /// Return u, v, r, mask
    std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor> Project(
            const core::Tensor& xyz,
            const core::Tensor& transformation =
                    core::Tensor::Eye(4, core::Dtype::Float64, core::Device()));

private:
    core::Tensor lidar_to_sensor_;

    core::Tensor azimuth_lut_;

    core::Tensor altitude_lut_;
    core::Tensor inv_altitude_lut_;

    core::Tensor unproj_dir_lut_;
    core::Tensor unproj_offset_lut_;
};

OdometryResult ComputeLiDAROdometryPointToPlane(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        // Note: currently target_normal_map is from point cloud
        const core::Tensor& target_normal_map,
        const LiDARCalib& calib,
        const core::Tensor& init_source_to_target,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
