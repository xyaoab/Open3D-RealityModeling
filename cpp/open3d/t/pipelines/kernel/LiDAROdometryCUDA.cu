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
        TransformIndexer proj_transform,
        TransformIndexer src2dst_transform,
        LiDARCalibConfig config,
        float* global_sum,
        float depth_diff) {
    // Find correspondence and obtain Jacobian at (x, y)
    // Note the built-in indexer uses (x, y) and (u, v) convention.

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

    if (y >= config.height || x >= config.width) return;

    float J[6] = {0}, reduction[21 + 6 + 2];
    float r = 0;
    bool valid = GetJacobianPointToPlane(
            source_vertex_indexer, source_mask_indexer, target_vertex_indexer,
            target_mask_indexer, target_normal_indexer, proj_transform,
            src2dst_transform, config, depth_diff, x, y, J, r);

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
        const LiDARCalibConfig& config,
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
    t::geometry::kernel::TransformIndexer proj_transform(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            (sensor_to_lidar.Matmul(init_source_to_target)).Contiguous());

    t::geometry::kernel::TransformIndexer src2dst_transform(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            init_source_to_target.Contiguous());

    // Result
    core::Tensor global_sum = core::Tensor::Zeros({29}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    // Launcher config
    const int kThreadSize = 16;
    const dim3 blocks((config.width + kThreadSize - 1) / kThreadSize,
                      (config.height + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputeLiDAROdometryPointToPlaneCUDAKernel<<<blocks, threads, 0,
                                                 core::cuda::GetStream()>>>(
            // Input
            source_vertex_indexer, source_mask_indexer, target_vertex_indexer,
            target_mask_indexer, target_normal_indexer,
            // Transform
            proj_transform, src2dst_transform,
            // LiDAR calib LUTs
            config,
            // Output
            global_sum_ptr,
            // Params
            depth_diff);
    core::cuda::Synchronize();

    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
