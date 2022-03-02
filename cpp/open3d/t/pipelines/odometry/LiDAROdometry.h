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
#include "open3d/t/geometry/LiDARImage.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

using t::geometry::LiDARImage;
using t::geometry::LiDARIntrinsic;

std::pair<OdometryResult, core::Tensor> LiDAROdometry(
        const LiDARImage& source,
        const LiDARImage& target,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float dist_diff,
        const OdometryConvergenceCriteria& criteria);

std::pair<OdometryResult, core::Tensor> LiDAROdometry(
        const LiDARImage& source,
        const LiDARImage& target,
        const core::Tensor& target_normal_map,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float dist_diff,
        const OdometryConvergenceCriteria& criteria);

std::pair<OdometryResult, core::Tensor> ComputeLiDAROdometryPointToPlane(
        const core::Tensor& source_vertex_map,
        const core::Tensor& source_mask_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        // Note: currently target_normal_map is from point cloud
        const core::Tensor& target_normal_map,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_diff);

std::pair<OdometryResult, core::Tensor> LiDAROdometryGNC(
        const LiDARImage& source,
        const LiDARImage& target,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float mu,
        const float depth_diff,
        const float division_factor,
        const int gnc_iterations,
        const OdometryConvergenceCriteria& criteria);

std::pair<OdometryResult, core::Tensor> LiDAROdometryGNC(
        const LiDARImage& source,
        const LiDARImage& target,
        const core::Tensor& target_normal_map,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float mu,
        const float depth_diff,
        const float division_factor,
        const int gnc_iterations,
        const OdometryConvergenceCriteria& criteria);

// In is_init=true, compute correspondences as initialization.
// In is_init=false, fix correspondences and update weights.
// The change of mu should be handled in the caller.
OdometryResult ComputeLiDAROdometryPointToPlaneGNC(
        const core::Tensor& source_vertex_map,
        const core::Tensor& source_mask_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        // Note: currently target_normal_map is from point cloud
        const core::Tensor& target_normal_map,
        core::Tensor& correspondences,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float mu,
        bool is_init);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
