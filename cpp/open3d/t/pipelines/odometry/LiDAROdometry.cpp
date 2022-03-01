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

std::pair<OdometryResult, core::Tensor> LiDAROdometry(
        const LiDARImage& source,
        const LiDARImage& target,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float dist_diff,
        const OdometryConvergenceCriteria& criteria) {
    core::Tensor target_normal_map = target.GetNormalMap(calib);
    return LiDAROdometry(source, target, target_normal_map, calib,
                         init_source_to_target, depth_min, depth_max, dist_diff,
                         criteria);
}

std::pair<OdometryResult, core::Tensor> LiDAROdometry(
        const LiDARImage& source,
        const LiDARImage& target,
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
            source.Unproject(calib, identity, depth_min, depth_max);
    std::tie(target_vertex_map, target_mask_map) =
            target.Unproject(calib, identity, depth_min, depth_max);

    auto init_trans = init_source_to_target.Clone();

    // Offset translation
    if (init_trans.AllClose(
                core::Tensor::Eye(4, core::Dtype::Float64, core::Device()))) {
        auto source_mean =
                source_vertex_map.IndexGet({source_mask_map}).Mean({0});
        auto target_mean =
                target_vertex_map.IndexGet({target_mask_map}).Mean({0});
        init_trans.SetItem({core::TensorKey::Slice(0, 3, 1),
                            core::TensorKey::Slice(3, 4, 1)},
                           (target_mean - source_mean).View({3, 1}));
    }

    OdometryResult result(init_trans);
    core::Tensor correspondences;
    for (int i = 0; i < criteria.max_iteration_; ++i) {
        auto res = ComputeLiDAROdometryPointToPlane(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, calib,
                result.transformation_, dist_diff);
        auto delta_result = res.first;
        correspondences = res.second;

        result.transformation_ =
                (delta_result.transformation_.Matmul(result.transformation_))
                        .Contiguous();
        utility::LogDebug("iter {}: rmse = {}, fitness = {}", i,
                          delta_result.inlier_rmse_, delta_result.fitness_);
    }

    return std::make_pair(result, correspondences);
}

std::pair<OdometryResult, core::Tensor> ComputeLiDAROdometryPointToPlane(
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

    core::Tensor correspondences;
    kernel::odometry::ComputeLiDAROdometryPointToPlane(
            source_vertex_map, source_mask_map, target_vertex_map,
            target_mask_map, target_normal_map, init_source_to_target,
            calib.sensor_to_lidar_, geometry::LiDARIntrinsicPtrs(calib),
            se3_delta, inlier_residual, inlier_count, depth_diff,
            correspondences);

    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }

    auto result = OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
    return std::make_pair(result, correspondences);
}

std::pair<OdometryResult, core::Tensor> LiDAROdometryGNC(
        const LiDARImage& source,
        const LiDARImage& target,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float mu,
        const float dist_diff,
        const float division_factor,
        const OdometryConvergenceCriteria& criteria) {
    core::Tensor target_normal_map = target.GetNormalMap(calib);
    return LiDAROdometryGNC(source, target, target_normal_map, calib,
                            init_source_to_target, depth_min, depth_max, mu,
                            dist_diff, division_factor, criteria);
}

std::pair<OdometryResult, core::Tensor> LiDAROdometryGNC(
        const LiDARImage& source,
        const LiDARImage& target,
        const core::Tensor& target_normal_map,
        const LiDARIntrinsic& calib,
        const core::Tensor& init_source_to_target,
        const float depth_min,
        const float depth_max,
        const float mu,
        const float dist_diff,
        const float division_factor,
        const OdometryConvergenceCriteria& criteria) {
    core::Tensor source_vertex_map, source_mask_map;
    core::Tensor target_vertex_map, target_mask_map;

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());
    std::tie(source_vertex_map, source_mask_map) =
            source.Unproject(calib, identity, depth_min, depth_max);
    std::tie(target_vertex_map, target_mask_map) =
            target.Unproject(calib, identity, depth_min, depth_max);

    auto init_trans = init_source_to_target.Clone();

    // Offset translation
    if (init_trans.AllClose(
                core::Tensor::Eye(4, core::Dtype::Float64, core::Device()))) {
        auto source_mean =
                source_vertex_map.IndexGet({source_mask_map}).Mean({0});
        auto target_mean =
                target_vertex_map.IndexGet({target_mask_map}).Mean({0});
        init_trans.SetItem({core::TensorKey::Slice(0, 3, 1),
                            core::TensorKey::Slice(3, 4, 1)},
                           (target_mean - source_mean).View({3, 1}));
    }

    OdometryResult result(init_trans);
    core::Tensor correspondences;
    auto delta_result = ComputeLiDAROdometryPointToPlaneGNC(
            source_vertex_map, source_mask_map, target_vertex_map,
            target_mask_map, target_normal_map, correspondences, calib,
            result.transformation_, mu, /*is_init=*/true);
    result.transformation_ =
            (delta_result.transformation_.Matmul(result.transformation_))
                    .Contiguous();

    float mu_curr = mu;
    for (int i = 0; i < criteria.max_iteration_; ++i) {
        delta_result = ComputeLiDAROdometryPointToPlaneGNC(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, correspondences, calib,
                result.transformation_, mu, /*is_init=*/false);
        result.transformation_ =
                (delta_result.transformation_.Matmul(result.transformation_))
                        .Contiguous();
        if (i % 4 == 0 && mu_curr > dist_diff) {
            mu_curr /= division_factor;
        }
        utility::LogDebug("iter {}: rmse = {}, fitness = {}", i,
                          delta_result.inlier_rmse_, delta_result.fitness_);
    }

    return std::make_pair(result, correspondences);
}

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
        bool is_init) {
    core::Tensor se3_delta;
    float inlier_residual;
    int inlier_count;

    kernel::odometry::ComputeLiDAROdometryPointToPlaneGNC(
            source_vertex_map, source_mask_map, target_vertex_map,
            target_mask_map, target_normal_map, correspondences,
            init_source_to_target, calib.sensor_to_lidar_,
            geometry::LiDARIntrinsicPtrs(calib), se3_delta, inlier_residual,
            inlier_count, mu, is_init);

    // Check inlier_count, source_vertex_map's shape is non-zero guaranteed.
    if (inlier_count <= 0) {
        utility::LogError("Invalid inlier_count value {}, must be > 0.",
                          inlier_count);
    }

    auto result = OdometryResult(
            pipelines::kernel::PoseToTransformation(se3_delta),
            inlier_residual / inlier_count,
            double(inlier_count) / double(source_vertex_map.GetShape(0) *
                                          source_vertex_map.GetShape(1)));
    return result;
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
