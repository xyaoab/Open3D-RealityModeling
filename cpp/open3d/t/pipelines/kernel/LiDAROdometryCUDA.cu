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

#include <cuda.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/LiDAROdometryImpl.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

using t::geometry::kernel::NDArrayIndexer;
using t::geometry::kernel::TransformIndexer;

__global__ void ComputeLiDAROdometryPointToPlaneCUDAKernel(
        NDArrayIndexer source_vertex_indexer,
        NDArrayIndexer source_mask_indexer,
        NDArrayIndexer target_vertex_indexer,
        NDArrayIndexer target_mask_indexer,
        NDArrayIndexer target_normal_indexer,
        TransformIndexer transform_indexer,
        TransformIndexer sensor_to_lidar_indexer,
        const float* azimuth_lut_ptr,
        const float* altitude_lut_ptr,
        const int64_t* inv_altitude_lut_ptr,
        float* global_sum,
        int64_t height,
        int64_t width,
        int64_t inv_lut_length,
        float azimuth_resolution,
        float azimuth_deg_to_pixel,
        float altitude_resolution,
        float altitude_min,
        float depth_diff) {
    // Lookup vi from altitude lut
    auto LookUpV = [=] OPEN3D_DEVICE(float phi_deg) -> int64_t {
        int64_t phi_int = static_cast<int64_t>(
                round((phi_deg - altitude_min) / altitude_resolution));
        if (phi_int < 0 || phi_int >= inv_lut_length) {
            return -1;
        }

        int64_t v0 = height - 1 - inv_altitude_lut_ptr[phi_int];
        int64_t v1 = max(v0 - 1, 0l);
        int64_t v2 = min(v0 + 1, height - 1);

        float diff0 = abs(altitude_lut_ptr[v0] - phi_deg);
        float diff1 = abs(altitude_lut_ptr[v1] - phi_deg);
        float diff2 = abs(altitude_lut_ptr[v2] - phi_deg);

        bool flag = diff0 < diff1;
        float diff = flag ? diff0 : diff1;
        int64_t v = flag ? v0 : v1;

        return diff < diff2 ? v : v2;
    };

    // Project xyz -> uvr
    auto DeviceProject = [=] OPEN3D_DEVICE(float x_in, float y_in, float z_in,
                                           int64_t* ui, int64_t* vi,
                                           float* r) -> bool {
        float x, y, z;
        sensor_to_lidar_indexer.RigidTransform(x_in, y_in, z_in, &x, &y, &z);
        *r = sqrt(x * x + y * y + z * z);

        // Estimate u
        float u = atan2(y, x);
        u = (u < 0) ? TWO_PI + u : u;
        u = TWO_PI - u;

        // Estimate v
        float phi = asin(z / *r);
        int64_t v = LookUpV(phi * RAD2DEG);

        if (v >= 0) {
            u = (u - azimuth_lut_ptr[v] * DEG2RAD) / azimuth_resolution;
            u = (u < 0) ? u + width : u;
            u = (u >= width) ? u - width : u;
            *ui = static_cast<int64_t>(u);
            *vi = static_cast<int64_t>(v);
            return true;
        } else {
            *ui = -1;
            *vi = -1;
            return false;
        }
    };

    // Find correspondence and obtain Jacobian at (x, y)
    // Note the built-in indexer uses (x, y) and (u, v) convention.
    auto GetJacobianPointToPlane = [=] OPEN3D_DEVICE(int x, int y, float* J_ij,
                                                     float& r) -> bool {
        float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
        bool mask_v = *source_mask_indexer.GetDataPtr<bool>(x, y);
        if (!mask_v) return false;

        // Transform source points to the target camera's coordinate space.
        float T_source_to_target_v[3];
        transform_indexer.RigidTransform(
                source_v[0], source_v[1], source_v[2], &T_source_to_target_v[0],
                &T_source_to_target_v[1], &T_source_to_target_v[2]);

        int64_t ui, vi;
        float d;
        bool mask_proj =
                DeviceProject(T_source_to_target_v[0], T_source_to_target_v[1],
                              T_source_to_target_v[2], &ui, &vi, &d);

        if (!mask_proj || !(*target_mask_indexer.GetDataPtr<bool>(ui, vi))) {
            return false;
        }

        float* target_v = target_vertex_indexer.GetDataPtr<float>(ui, vi);
        float* target_n = target_normal_indexer.GetDataPtr<float>(ui, vi);

        r = (T_source_to_target_v[0] - target_v[0]) * target_n[0] +
            (T_source_to_target_v[1] - target_v[1]) * target_n[1] +
            (T_source_to_target_v[2] - target_v[2]) * target_n[2];

        // Pseudo huber loss

        // float w = abs(r) > depth_diff ? 0 : 1;
        float depth_diff2 = depth_diff * depth_diff;
        float w = 1.0 / (depth_diff2 * sqrt((r * r / depth_diff2) + 1));

        J_ij[0] = w * (-T_source_to_target_v[2] * target_n[1] +
                       T_source_to_target_v[1] * target_n[2]);
        J_ij[1] = w * (T_source_to_target_v[2] * target_n[0] -
                       T_source_to_target_v[0] * target_n[2]);
        J_ij[2] = w * (-T_source_to_target_v[1] * target_n[0] +
                       T_source_to_target_v[0] * target_n[1]);
        J_ij[3] = w * target_n[0];
        J_ij[4] = w * target_n[1];
        J_ij[5] = w * target_n[2];
        r = w * r;

        return true;
    };

    const int kBlockSize = 256;
    __shared__ float local_sum0[kBlockSize];
    __shared__ float local_sum1[kBlockSize];
    __shared__ float local_sum2[kBlockSize];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (y >= height || x >= width) return;

    float J[6] = {0}, reduction[21 + 6 + 2];
    float r = 0;
    bool valid = GetJacobianPointToPlane(x, y, J, r);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J[i] * J[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J[i] * r;
    }
    reduction[offset++] = r * r;
    reduction[offset++] = valid;

    // Sum reduction: JtJ(21) and Jtr(6)
    for (size_t i = 0; i < 27; i += 3) {
        local_sum0[tid] = valid ? reduction[i + 0] : 0;
        local_sum1[tid] = valid ? reduction[i + 1] : 0;
        local_sum2[tid] = valid ? reduction[i + 2] : 0;
        __syncthreads();

        BlockReduceSum<float, kBlockSize>(tid, local_sum0, local_sum1,
                                          local_sum2);

        if (tid == 0) {
            atomicAdd(&global_sum[i + 0], local_sum0[0]);
            atomicAdd(&global_sum[i + 1], local_sum1[0]);
            atomicAdd(&global_sum[i + 2], local_sum2[0]);
        }
        __syncthreads();
    }

    // Sum reduction: residual(1) and inlier(1)
    {
        local_sum0[tid] = valid ? reduction[27] : 0;
        local_sum1[tid] = valid ? reduction[28] : 0;
        __syncthreads();

        BlockReduceSum<float, kBlockSize>(tid, local_sum0, local_sum1);
        if (tid == 0) {
            atomicAdd(&global_sum[27], local_sum0[0]);
            atomicAdd(&global_sum[28], local_sum1[0]);
        }
        __syncthreads();
    }
}

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
        float depth_diff) {
    core::Device device = source_vertex_map.GetDevice();

    // Index source data
    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer source_mask_indexer(source_mask_map, 2);

    // Index target data
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);
    NDArrayIndexer target_mask_indexer(target_mask_map, 2);
    NDArrayIndexer target_normal_indexer(target_normal_map, 2);

    // Wrap transformation
    t::geometry::kernel::TransformIndexer sensor_to_lidar_indexer(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            sensor_to_lidar.Contiguous());

    t::geometry::kernel::TransformIndexer transform_indexer(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            init_source_to_target.Contiguous());

    // Projection LUTs
    const float* azimuth_lut_ptr = azimuth_lut.GetDataPtr<float>();
    const float* altitude_lut_ptr = altitude_lut.GetDataPtr<float>();
    const int64_t* inv_altitude_lut_ptr =
            inv_altitude_lut.GetDataPtr<int64_t>();

    // Projection consts
    const int64_t width = 1024;
    const int64_t height = 128;
    const int64_t inv_lut_length = inv_altitude_lut.GetLength();

    const float azimuth_resolution = TWO_PI / width;
    const float azimuth_deg_to_pixel = width / 360.0;
    const float altitude_resolution = 0.4;
    const float altitude_min = altitude_lut[height - 1].Item<float>();

    // Result
    core::Tensor global_sum = core::Tensor::Zeros({29}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    // Launcher config
    const int kThreadSize = 16;
    const dim3 blocks((width + kThreadSize - 1) / kThreadSize,
                      (height + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputeLiDAROdometryPointToPlaneCUDAKernel<<<blocks, threads, 0,
                                                 core::cuda::GetStream()>>>(
            // Input
            source_vertex_indexer, source_mask_indexer, target_vertex_indexer,
            target_mask_indexer, target_normal_indexer,
            // Transform
            transform_indexer, sensor_to_lidar_indexer,
            // LiDAR calib LUTs
            azimuth_lut_ptr, altitude_lut_ptr, inv_altitude_lut_ptr,
            // Output
            global_sum_ptr,
            // Params
            height, width, inv_lut_length, azimuth_resolution,
            azimuth_deg_to_pixel, altitude_resolution, altitude_min,
            depth_diff);
    core::cuda::Synchronize();

    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
