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

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void LiDARUnproject(const core::Tensor& range_image,
                    const core::Tensor& transformation,
                    const core::Tensor& dir_lut,
                    const core::Tensor& offset_lut,
                    core::Tensor& xyz_im,
                    core::Tensor& mask_im,
                    float depth_scale,
                    float depth_min,
                    float depth_max);

void LiDARProject(const core::Tensor& xyz,
                  const core::Tensor& transformation,
                  const core::Tensor& azimuth_lut,
                  const core::Tensor& altitude_lut,
                  const core::Tensor& inv_altitude_lut,
                  core::Tensor& u,
                  core::Tensor& v,
                  core::Tensor& r,
                  core::Tensor& mask);

void ComputeLiDAROdometryPointToPlane(
        // source input
        const core::Tensor& source_vertex_map,
        const core::Tensor& source_mask_map,
        // target input
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        const core::Tensor& target_normal_map,
        // init transformation
        const core::Tensor& init_source_to_target,
        const core::Tensor& sensor_to_lidar,
        // LiDAR calibration
        const core::Tensor& azimuth_lut,
        const core::Tensor& altitude_lut,
        const core::Tensor& inv_altitude_lut,
        // Output linear system result
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        // Other params
        float depth_diff);

void LiDARUnprojectCPU(const core::Tensor& range_image,
                       const core::Tensor& transformation,
                       const core::Tensor& dir_lut,
                       const core::Tensor& offset_lut,
                       core::Tensor& xyz_im,
                       core::Tensor& mask_im,
                       float depth_scale,
                       float depth_min,
                       float depth_max);

void LiDARProjectCPU(const core::Tensor& xyz,
                     const core::Tensor& transformation,
                     const core::Tensor& azimuth_lut,
                     const core::Tensor& altitude_lut,
                     const core::Tensor& inv_altitude_lut,
                     core::Tensor& u,
                     core::Tensor& v,
                     core::Tensor& r,
                     core::Tensor& mask);

#ifdef BUILD_CUDA_MODULE
void LiDARUnprojectCUDA(const core::Tensor& range_image,
                        const core::Tensor& transformation,
                        const core::Tensor& dir_lut,
                        const core::Tensor& offset_lut,
                        core::Tensor& xyz_im,
                        core::Tensor& mask_im,
                        float depth_scale,
                        float depth_min,
                        float depth_max);

void LiDARProjectCUDA(const core::Tensor& xyz,
                      const core::Tensor& transformation,
                      const core::Tensor& azimuth_lut,
                      const core::Tensor& altitude_lut,
                      const core::Tensor& inv_altitude_lut,
                      core::Tensor& u,
                      core::Tensor& v,
                      core::Tensor& r,
                      core::Tensor& mask);

void ComputeLiDAROdometryPointToPlaneCUDA(
        // source input
        const core::Tensor& source_vertex_map,
        const core::Tensor& source_mask_map,
        // target input
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_mask_map,
        const core::Tensor& target_normal_map,
        // init transformation
        const core::Tensor& init_source_to_target,
        const core::Tensor& sensor_to_lidar,
        // LiDAR calibration
        const core::Tensor& azimuth_lut,
        const core::Tensor& altitude_lut,
        const core::Tensor& inv_altitude_lut,
        // Output linear system result
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        // Other params
        float depth_diff);

#endif

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
