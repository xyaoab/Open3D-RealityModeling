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

#include <atomic>
#include <cmath>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

using index_t = int;
using ArrayIndexer = TArrayIndexer<index_t>;

template <typename tsdf_t, typename weight_t, typename color_t>
#if defined(__CUDACC__)
void RaySampleCUDA
#else
void RaySampleCPU
#endif
        (std::shared_ptr<core::HashMap>& hashmap,
         const TensorMap& block_value_map,
         TensorMap& renderings_map,
         const core::Tensor& rays_o,
         const core::Tensor& rays_d,
         index_t samples,
         index_t block_resolution,
         float voxel_size,
         float depth_scale,
         float depth_min,
         float depth_max,
         float weight_threshold,
         float trunc_voxel_multiplier) {
    using Key = utility::MiniVec<index_t, 3>;
    using Hash = utility::MiniVecHash<index_t, 3>;
    using Eq = utility::MiniVecEq<index_t, 3>;

    auto device_hashmap = hashmap->GetDeviceHashBackend();
#if defined(__CUDACC__)
    auto cuda_hashmap =
            std::dynamic_pointer_cast<core::StdGPUHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cuda_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CUDA raycasting only supports STDGPU.");
    }
    auto hashmap_impl = cuda_hashmap->GetImpl();
#else
    auto cpu_hashmap =
            std::dynamic_pointer_cast<core::TBBHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cpu_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CPU raycasting only supports TBB.");
    }
    auto hashmap_impl = *cpu_hashmap->GetImpl();
#endif

#ifndef __CUDACC__
    using std::max;
    using std::min;
#endif

    core::Device device = hashmap->GetDevice();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    const tsdf_t* tsdf_base_ptr =
            block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();

    ArrayIndexer rays_o_indexer(rays_o, 1);
    ArrayIndexer rays_d_indexer(rays_d, 1);

    // Geometry
    ArrayIndexer mask_indexer(renderings_map.at("mask"), 1);
    ArrayIndexer depth_indexer(renderings_map.at("depth"), 2);
    ArrayIndexer weight_indexer(renderings_map.at("weight"), 2);

    // Diff rendering
    ArrayIndexer index_indexer(renderings_map.at("index"), 2);
    ArrayIndexer interp_ratio_indexer(renderings_map.at("interp_ratio"), 2);
    ArrayIndexer interp_ratio_dx_indexer(renderings_map.at("interp_ratio_dx"),
                                         2);
    ArrayIndexer interp_ratio_dy_indexer(renderings_map.at("interp_ratio_dy"),
                                         2);
    ArrayIndexer interp_ratio_dz_indexer(renderings_map.at("interp_ratio_dz"),
                                         2);

    index_t n = rays_o.GetLength();

    float block_size = voxel_size * block_resolution;
    index_t resolution2 = block_resolution * block_resolution;
    index_t resolution3 = resolution2 * block_resolution;

#ifndef __CUDACC__
    using std::max;
    using std::sqrt;
#endif

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        // Helper: access buf index for certain block<int> and voxel<int>
        // coordinates
        auto GetLinearIdxAtP = [&] OPEN3D_DEVICE(
                                       index_t x_b, index_t y_b, index_t z_b,
                                       index_t x_v, index_t y_v, index_t z_v,
                                       core::buf_index_t block_buf_idx,
                                       MiniVecCache & cache) -> index_t {
            index_t x_vn = (x_v + block_resolution) % block_resolution;
            index_t y_vn = (y_v + block_resolution) % block_resolution;
            index_t z_vn = (z_v + block_resolution) % block_resolution;

            index_t dx_b = Sign(x_v - x_vn);
            index_t dy_b = Sign(y_v - y_vn);
            index_t dz_b = Sign(z_v - z_vn);

            if (dx_b == 0 && dy_b == 0 && dz_b == 0) {
                return block_buf_idx * resolution3 + z_v * resolution2 +
                       y_v * block_resolution + x_v;
            } else {
                Key key(x_b + dx_b, y_b + dy_b, z_b + dz_b);

                index_t block_buf_idx = cache.Check(key[0], key[1], key[2]);
                if (block_buf_idx < 0) {
                    auto iter = hashmap_impl.find(key);
                    if (iter == hashmap_impl.end()) return -1;
                    block_buf_idx = iter->second;
                    cache.Update(key[0], key[1], key[2], block_buf_idx);
                }

                return block_buf_idx * resolution3 + z_vn * resolution2 +
                       y_vn * block_resolution + x_vn;
            }
        };

        // Helper: access buf index for voxel with point on a ray<float> after
        // floor op
        auto GetLinearIdxAtT = [&] OPEN3D_DEVICE(
                                       float x_o, float y_o, float z_o,
                                       float x_d, float y_d, float z_d, float t,
                                       MiniVecCache& cache) -> index_t {
            float x_g = x_o + t * x_d;
            float y_g = y_o + t * y_d;
            float z_g = z_o + t * z_d;

            // MiniVec coordinate and look up
            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));

            Key key(x_b, y_b, z_b);
            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                if (iter == hashmap_impl.end()) return -1;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            // Voxel coordinate and look up
            index_t x_v = index_t((x_g - x_b * block_size) / voxel_size);
            index_t y_v = index_t((y_g - y_b * block_size) / voxel_size);
            index_t z_v = index_t((z_g - z_b * block_size) / voxel_size);

            return block_buf_idx * resolution3 + z_v * resolution2 +
                   y_v * block_resolution + x_v;
        };

        // Iterative ray intersection check
        float* ray_o_ptr = rays_o_indexer.GetDataPtr<float>(workload_idx);
        float* ray_d_ptr = rays_d_indexer.GetDataPtr<float>(workload_idx);

        float x_o = ray_o_ptr[0];
        float y_o = ray_o_ptr[1];
        float z_o = ray_o_ptr[2];

        float x_d = ray_d_ptr[0];
        float y_d = ray_d_ptr[1];
        float z_d = ray_d_ptr[2];

        float x_g = 0, y_g = 0, z_g = 0;

        MiniVecCache cache{0, 0, 0, -1};

        // Helper: sample
        auto Sample = [&] OPEN3D_DEVICE(float t_intersect, int& sample_cnt) {
            x_g = x_o + t_intersect * x_d;
            y_g = y_o + t_intersect * y_d;
            z_g = z_o + t_intersect * z_d;

            // Trivial depth assignment
            *depth_indexer.GetDataPtr<float>(workload_idx, sample_cnt) =
                    t_intersect * depth_scale;

            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));
            float x_v = (x_g - float(x_b) * block_size) / voxel_size;
            float y_v = (y_g - float(y_b) * block_size) / voxel_size;
            float z_v = (z_g - float(z_b) * block_size) / voxel_size;
            Key key(x_b, y_b, z_b);

            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                // Current block not allocated, return
                if (iter == hashmap_impl.end()) return;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            index_t x_v_floor = static_cast<index_t>(floorf(x_v));
            index_t y_v_floor = static_cast<index_t>(floorf(y_v));
            index_t z_v_floor = static_cast<index_t>(floorf(z_v));

            float ratio_x = x_v - float(x_v_floor);
            float ratio_y = y_v - float(y_v_floor);
            float ratio_z = z_v - float(z_v_floor);

            auto index_ptr =
                    index_indexer.GetDataPtr<int64_t>(workload_idx, sample_cnt);
            auto interp_ratio_ptr = interp_ratio_indexer.GetDataPtr<float>(
                    workload_idx, sample_cnt);
            auto interp_ratio_dx_ptr =
                    interp_ratio_dx_indexer.GetDataPtr<float>(workload_idx,
                                                              sample_cnt);
            auto interp_ratio_dy_ptr =
                    interp_ratio_dy_indexer.GetDataPtr<float>(workload_idx,
                                                              sample_cnt);
            auto interp_ratio_dz_ptr =
                    interp_ratio_dz_indexer.GetDataPtr<float>(workload_idx,
                                                              sample_cnt);

            // Assign indices and interp ratios at 8 neighbors
            float w = 0;
            for (index_t k = 0; k < 8; ++k) {
                index_t dx_v = (k & 1) > 0 ? 1 : 0;
                index_t dy_v = (k & 2) > 0 ? 1 : 0;
                index_t dz_v = (k & 4) > 0 ? 1 : 0;

                index_t linear_idx_k = GetLinearIdxAtP(
                        x_b, y_b, z_b, x_v_floor + dx_v, y_v_floor + dy_v,
                        z_v_floor + dz_v, block_buf_idx, cache);

                if (linear_idx_k >= 0 && weight_base_ptr[linear_idx_k] > 0) {
                    float rx = dx_v * (ratio_x) + (1 - dx_v) * (1 - ratio_x);
                    float ry = dy_v * (ratio_y) + (1 - dy_v) * (1 - ratio_y);
                    float rz = dz_v * (ratio_z) + (1 - dz_v) * (1 - ratio_z);
                    float r = rx * ry * rz;

                    float interp_ratio_dx = ry * rz * (2 * dx_v - 1);
                    float interp_ratio_dy = rx * rz * (2 * dy_v - 1);
                    float interp_ratio_dz = rx * ry * (2 * dz_v - 1);

                    index_ptr[k] = linear_idx_k;
                    interp_ratio_ptr[k] = r;
                    interp_ratio_dx_ptr[k] = interp_ratio_dx;
                    interp_ratio_dy_ptr[k] = interp_ratio_dy;
                    interp_ratio_dz_ptr[k] = interp_ratio_dz;

                    w += r * weight_base_ptr[linear_idx_k];
                }
            }  // loop over 8 neighbors
            *weight_indexer.GetDataPtr<float>(workload_idx, sample_cnt) = w;

            sample_cnt += 1;
        };

        float t = depth_min;
        float t_max = depth_max;
        if (t >= t_max) return;

        // Search for min range
        index_t linear_idx = -1;
        while (t < t_max && linear_idx < 0) {
            linear_idx =
                    GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);
            t += voxel_size;
        }
        float t_min = t;

        // Search for max range
        // Allow failure for 3 consecutive blocks
        int outbound_cnt = 0;
        while (outbound_cnt < 3) {
            linear_idx =
                    GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);

            if (linear_idx >= 0) {
                outbound_cnt = 0;
            } else {
                if (outbound_cnt == 0) {
                    t_max = t;
                }
                outbound_cnt++;
            }
            t += block_size;
        }

        // Search for the surface
        float t_intersect = -1;
        float t_prev = t;

        float tsdf_prev = -1.0f;
        float tsdf = 1.0;
        float sdf_trunc = voxel_size * trunc_voxel_multiplier;
        float w = 0.0;

        t = t_min;
        while (t < t_max) {
            index_t linear_idx =
                    GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);

            if (linear_idx < 0) {
                t_prev = t;
                t += block_size * 0.5;
            } else {
                tsdf_prev = tsdf;
                tsdf = tsdf_base_ptr[linear_idx];
                w = weight_base_ptr[linear_idx];

                if (tsdf_prev > 0 && w >= weight_threshold && tsdf <= 0) {
                    t_intersect = (t * tsdf_prev - t_prev * tsdf) /
                                  (tsdf_prev - tsdf);
                    break;
                }
                t_prev = t;
                float delta = tsdf * sdf_trunc;
                t += delta < voxel_size ? voxel_size : delta;
            }
        }

        // Uniform sample
        int sample_cnt = 0;
        if (t_intersect < 0) {
            float step = (t_max - t_min) / samples;
            if (step <= 0) return;
            t = t_min;
            for (int s = 0; s < samples; ++s, t += step) {
                index_t linear_idx =
                        GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);
                if (linear_idx >= 0) {
                    Sample(t, sample_cnt);
                }
            }
        } else {  // Sample around surfaces
            *mask_indexer.GetDataPtr<bool>(workload_idx) = true;
            float step = (sdf_trunc * 2) / samples;
            t = t_intersect - sdf_trunc;
            for (int s = 0; s < samples; ++s, t += step) {
                index_t linear_idx =
                        GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);
                if (linear_idx >= 0) {
                    Sample(t, sample_cnt);
                }
            }
        }
    });

#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

template <typename tsdf_t, typename weight_t, typename color_t>
#if defined(__CUDACC__)
void RayMarchCUDA
#else
void RayMarchCPU
#endif
        (std::shared_ptr<core::HashMap>& hashmap,
         const TensorMap& block_value_map,
         TensorMap& renderings_map,
         const core::Tensor& intrinsic,
         const core::Tensor& extrinsics,
         index_t h,
         index_t w,
         index_t samples,
         index_t block_resolution,
         float voxel_size,
         float depth_scale,
         float depth_min,
         float depth_max,
         float weight_threshold,
         float trunc_voxel_multiplier) {
    using Key = utility::MiniVec<index_t, 3>;
    using Hash = utility::MiniVecHash<index_t, 3>;
    using Eq = utility::MiniVecEq<index_t, 3>;

    auto device_hashmap = hashmap->GetDeviceHashBackend();
#if defined(__CUDACC__)
    auto cuda_hashmap =
            std::dynamic_pointer_cast<core::StdGPUHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cuda_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CUDA raycasting only supports STDGPU.");
    }
    auto hashmap_impl = cuda_hashmap->GetImpl();
#else
    auto cpu_hashmap =
            std::dynamic_pointer_cast<core::TBBHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cpu_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CPU raycasting only supports TBB.");
    }
    auto hashmap_impl = *cpu_hashmap->GetImpl();
#endif

#ifndef __CUDACC__
    using std::max;
    using std::min;
#endif

    core::Device device = hashmap->GetDevice();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    const tsdf_t* tsdf_base_ptr =
            block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();

    // Geometry
    ArrayIndexer depth_indexer = ArrayIndexer(renderings_map.at("depth"), 3);

    // Diff rendering
    ArrayIndexer index_indexer = ArrayIndexer(renderings_map.at("index"), 3);
    ArrayIndexer interp_ratio_indexer =
            ArrayIndexer(renderings_map.at("interp_ratio"), 3);
    ArrayIndexer interp_ratio_dx_indexer =
            ArrayIndexer(renderings_map.at("interp_ratio_dx"), 3);
    ArrayIndexer interp_ratio_dy_indexer =
            ArrayIndexer(renderings_map.at("interp_ratio_dy"), 3);
    ArrayIndexer interp_ratio_dz_indexer =
            ArrayIndexer(renderings_map.at("interp_ratio_dz"), 3);

    TransformIndexer c2w_transform_indexer(
            intrinsic, t::geometry::InverseTransformation(extrinsics));
    TransformIndexer w2c_transform_indexer(intrinsic, extrinsics);

    index_t rows = h;
    index_t cols = w;
    index_t n = rows * cols;

    float block_size = voxel_size * block_resolution;
    index_t resolution2 = block_resolution * block_resolution;
    index_t resolution3 = resolution2 * block_resolution;

#ifndef __CUDACC__
    using std::max;
    using std::sqrt;
#endif

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        auto GetLinearIdxAtP = [&] OPEN3D_DEVICE(
                                       index_t x_b, index_t y_b, index_t z_b,
                                       index_t x_v, index_t y_v, index_t z_v,
                                       core::buf_index_t block_buf_idx,
                                       MiniVecCache & cache) -> index_t {
            index_t x_vn = (x_v + block_resolution) % block_resolution;
            index_t y_vn = (y_v + block_resolution) % block_resolution;
            index_t z_vn = (z_v + block_resolution) % block_resolution;

            index_t dx_b = Sign(x_v - x_vn);
            index_t dy_b = Sign(y_v - y_vn);
            index_t dz_b = Sign(z_v - z_vn);

            if (dx_b == 0 && dy_b == 0 && dz_b == 0) {
                return block_buf_idx * resolution3 + z_v * resolution2 +
                       y_v * block_resolution + x_v;
            } else {
                Key key(x_b + dx_b, y_b + dy_b, z_b + dz_b);

                index_t block_buf_idx = cache.Check(key[0], key[1], key[2]);
                if (block_buf_idx < 0) {
                    auto iter = hashmap_impl.find(key);
                    if (iter == hashmap_impl.end()) return -1;
                    block_buf_idx = iter->second;
                    cache.Update(key[0], key[1], key[2], block_buf_idx);
                }

                return block_buf_idx * resolution3 + z_vn * resolution2 +
                       y_vn * block_resolution + x_vn;
            }
        };

        auto GetLinearIdxAtT = [&] OPEN3D_DEVICE(
                                       float x_o, float y_o, float z_o,
                                       float x_d, float y_d, float z_d, float t,
                                       MiniVecCache& cache) -> index_t {
            float x_g = x_o + t * x_d;
            float y_g = y_o + t * y_d;
            float z_g = z_o + t * z_d;

            // MiniVec coordinate and look up
            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));

            Key key(x_b, y_b, z_b);
            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                if (iter == hashmap_impl.end()) return -1;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            // Voxel coordinate and look up
            index_t x_v = index_t((x_g - x_b * block_size) / voxel_size);
            index_t y_v = index_t((y_g - y_b * block_size) / voxel_size);
            index_t z_v = index_t((z_g - z_b * block_size) / voxel_size);

            return block_buf_idx * resolution3 + z_v * resolution2 +
                   y_v * block_resolution + x_v;
        };

        index_t y = workload_idx / cols;
        index_t x = workload_idx % cols;

        float t = depth_min;
        const float t_max = depth_max;
        if (t >= t_max) return;

        // Coordinates in camera and global
        float x_c = 0, y_c = 0, z_c = 0;
        float x_g = 0, y_g = 0, z_g = 0;
        float x_o = 0, y_o = 0, z_o = 0;

        // Iterative ray intersection check
        float t_prev = t;

        float tsdf_prev = -1.0f;
        float tsdf = 1.0;
        float sdf_trunc = voxel_size * trunc_voxel_multiplier;
        float w = 0.0;

        // Camera origin
        c2w_transform_indexer.RigidTransform(0, 0, 0, &x_o, &y_o, &z_o);

        // Direction
        c2w_transform_indexer.Unproject(static_cast<float>(x),
                                        static_cast<float>(y), 1.0f, &x_c, &y_c,
                                        &z_c);
        c2w_transform_indexer.RigidTransform(x_c, y_c, z_c, &x_g, &y_g, &z_g);
        float x_d = (x_g - x_o);
        float y_d = (y_g - y_o);
        float z_d = (z_g - z_o);

        MiniVecCache cache{0, 0, 0, -1};

        auto UpdateResult = [&] OPEN3D_DEVICE(float t_intersect,
                                              int& sample_cnt) {
            x_g = x_o + t_intersect * x_d;
            y_g = y_o + t_intersect * y_d;
            z_g = z_o + t_intersect * z_d;

            // Trivial vertex assignment
            *depth_indexer.GetDataPtr<float>(x, y, sample_cnt) =
                    t_intersect * depth_scale;

            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));
            float x_v = (x_g - float(x_b) * block_size) / voxel_size;
            float y_v = (y_g - float(y_b) * block_size) / voxel_size;
            float z_v = (z_g - float(z_b) * block_size) / voxel_size;

            Key key(x_b, y_b, z_b);

            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                // Current block not allocated, return
                if (iter == hashmap_impl.end()) return;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            index_t x_v_floor = static_cast<index_t>(floorf(x_v));
            index_t y_v_floor = static_cast<index_t>(floorf(y_v));
            index_t z_v_floor = static_cast<index_t>(floorf(z_v));

            float ratio_x = x_v - float(x_v_floor);
            float ratio_y = y_v - float(y_v_floor);
            float ratio_z = z_v - float(z_v_floor);

            auto index_ptr =
                    index_indexer.GetDataPtr<int64_t>(x, y, sample_cnt);
            auto interp_ratio_ptr =
                    interp_ratio_indexer.GetDataPtr<float>(x, y, sample_cnt);
            auto interp_ratio_dx_ptr =
                    interp_ratio_dx_indexer.GetDataPtr<float>(x, y, sample_cnt);
            auto interp_ratio_dy_ptr =
                    interp_ratio_dy_indexer.GetDataPtr<float>(x, y, sample_cnt);
            auto interp_ratio_dz_ptr =
                    interp_ratio_dz_indexer.GetDataPtr<float>(x, y, sample_cnt);

            for (index_t k = 0; k < 8; ++k) {
                index_t dx_v = (k & 1) > 0 ? 1 : 0;
                index_t dy_v = (k & 2) > 0 ? 1 : 0;
                index_t dz_v = (k & 4) > 0 ? 1 : 0;

                index_t linear_idx_k = GetLinearIdxAtP(
                        x_b, y_b, z_b, x_v_floor + dx_v, y_v_floor + dy_v,
                        z_v_floor + dz_v, block_buf_idx, cache);

                if (linear_idx_k >= 0 && weight_base_ptr[linear_idx_k] > 0) {
                    float rx = dx_v * (ratio_x) + (1 - dx_v) * (1 - ratio_x);
                    float ry = dy_v * (ratio_y) + (1 - dy_v) * (1 - ratio_y);
                    float rz = dz_v * (ratio_z) + (1 - dz_v) * (1 - ratio_z);
                    float r = rx * ry * rz;

                    float interp_ratio_dx = ry * rz * (2 * dx_v - 1);
                    float interp_ratio_dy = rx * rz * (2 * dy_v - 1);
                    float interp_ratio_dz = rx * ry * (2 * dz_v - 1);

                    index_ptr[k] = linear_idx_k;
                    interp_ratio_ptr[k] = r;
                    interp_ratio_dx_ptr[k] = interp_ratio_dx;
                    interp_ratio_dy_ptr[k] = interp_ratio_dy;
                    interp_ratio_dz_ptr[k] = interp_ratio_dz;
                }
            }  // loop over 8 neighbors

            sample_cnt += 1;
        };

        int sample_cnt = 0;
        while (t < t_max && sample_cnt < samples) {
            index_t linear_idx =
                    GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);

            if (linear_idx < 0) {
                t_prev = t;
                t += block_size;
            } else {
                tsdf_prev = tsdf;
                tsdf = tsdf_base_ptr[linear_idx];
                w = weight_base_ptr[linear_idx];

                if (tsdf_prev > 0 && w >= weight_threshold && tsdf <= 0) {
                    float t_intersect = (t * tsdf_prev - t_prev * tsdf) /
                                        (tsdf_prev - tsdf);

                    UpdateResult(t_intersect, sample_cnt);
                }
                t_prev = t;
                float delta = tsdf * sdf_trunc;
                t += delta < voxel_size ? voxel_size : delta;
            }
        }
    });

#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}
}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
