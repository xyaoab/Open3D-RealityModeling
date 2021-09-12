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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/LiDAROdometryImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeLiDAROdometryPointToPlaneCPU(
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

    int64_t n = config.width * config.height;
    std::vector<float> A_1x29(29, 0.0);
#ifdef _MSC_VER
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    int y = workload_idx / config.width;
                    int x = workload_idx % config.width;

                    float J_ij[6];
                    float r;
                    bool valid = GetJacobianPointToPlane(
                            source_vertex_indexer, source_mask_indexer,
                            target_vertex_indexer, target_mask_indexer,
                            target_normal_indexer, proj_transform,
                            src2dst_transform, config, depth_diff, x, y, J_ij,
                            r);

                    if (valid) {
                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] += J_ij[j] * J_ij[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_ij[j] * r;
                        }
                        A_reduction[27] += r * r;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _MSC_VER
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<float> a, std::vector<float> b) {
                std::vector<float> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif
    core::Tensor global_sum(A_1x29, {1, 29}, core::Float32, device);
    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
