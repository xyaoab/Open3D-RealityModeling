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

#include "open3d/t/pipelines/kernel/LiDAROdometry.h"

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void LiDARUnproject(const core::Tensor& range_image,
                    const core::Tensor& transformation,
                    const LiDARCalibConfig& config,
                    core::Tensor& xyz_im,
                    core::Tensor& mask_im,
                    float depth_scale,
                    float depth_min,
                    float depth_max) {
    core::Device device = range_image.GetDevice();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        LiDARUnprojectCPU(range_image, transformation, config, xyz_im, mask_im,
                          depth_scale, depth_min, depth_max);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(LiDARUnprojectCUDA, range_image, transformation, config,
                  xyz_im, mask_im, depth_scale, depth_min, depth_max);
    } else {
        utility::LogError("Unimplemented device {}", device.ToString());
    }
}

void LiDARProject(const core::Tensor& xyz,
                  const core::Tensor& transformation,
                  const LiDARCalibConfig& config,
                  core::Tensor& u,
                  core::Tensor& v,
                  core::Tensor& r,
                  core::Tensor& mask) {
    core::Device device = xyz.GetDevice();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        LiDARProjectCPU(xyz, transformation, config, u, v, r, mask);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(LiDARProjectCUDA, xyz, transformation, config, u, v, r, mask);
    } else {
        utility::LogError("Unimplemented device {}", device.ToString());
    }
}

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
        const LiDARCalibConfig& config,
        // Output linear system result
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        // Other params
        float depth_diff) {
    core::Device device = source_vertex_map.GetDevice();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        ComputeLiDAROdometryPointToPlaneCPU(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, init_source_to_target,
                sensor_to_lidar, config, delta, inlier_residual, inlier_count,
                depth_diff);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        ComputeLiDAROdometryPointToPlaneCUDA(
                source_vertex_map, source_mask_map, target_vertex_map,
                target_mask_map, target_normal_map, init_source_to_target,
                sensor_to_lidar, config, delta, inlier_residual, inlier_count,
                depth_diff);
    } else {
        utility::LogError("Unimplemented device {}", device.ToString());
    }
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
