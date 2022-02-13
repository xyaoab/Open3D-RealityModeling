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

// Private header. Do not include in Open3d.h.

#include <cmath>

#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/odometry/LiDAROdometry.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

#define PI 3.14159265358979323846
#define TWO_PI (2 * PI)
#define RAD2DEG (180 / PI)
#define DEG2RAD (PI / 180)
using t::geometry::LiDARIntrinsicPtrs;
using t::geometry::kernel::NDArrayIndexer;
using t::geometry::kernel::TransformIndexer;

#ifndef __CUDACC__
using std::abs;
using std::asin;
using std::atan2;
using std::max;
using std::min;
using std::round;
using std::sqrt;
#endif

inline OPEN3D_DEVICE int64_t LookUpV(const LiDARIntrinsicPtrs& config,
                                     float phi_deg) {
    int64_t phi_int =
            static_cast<int64_t>(round((phi_deg - config.min_altitude) /
                                       config.inv_altitude_lut_resolution));
    if (phi_int < 0 || phi_int >= config.inv_altitude_lut_length) {
        return -1;
    }

    int64_t v0 = config.height - 1 - config.inv_altitude_lut_ptr[phi_int];
    int64_t v1 = max(v0 - 1, 0l);
    int64_t v2 = min(v0 + 1, config.height - 1);

    float diff0 = abs(config.altitude_lut_ptr[v0] - phi_deg);
    float diff1 = abs(config.altitude_lut_ptr[v1] - phi_deg);
    float diff2 = abs(config.altitude_lut_ptr[v2] - phi_deg);

    bool flag = diff0 < diff1;
    float diff = flag ? diff0 : diff1;
    int64_t v = flag ? v0 : v1;

    return diff < diff2 ? v : v2;
};

inline OPEN3D_DEVICE bool DeviceProjectLUT(
        const LiDARIntrinsicPtrs& config,
        // this transform indexer should be nested:
        // sensor_to_lidar @ rigid_transformation
        const TransformIndexer& transform_indexer,
        float x_in,
        float y_in,
        float z_in,
        int64_t* ui,
        int64_t* vi,
        float* r) {
    float x, y, z;
    transform_indexer.RigidTransform(x_in, y_in, z_in, &x, &y, &z);
    *r = sqrt(x * x + y * y + z * z);

    // Estimate u
    float u = atan2(y, x);
    u = (u < 0) ? TWO_PI + u : u;
    u = TWO_PI - u;

    // Estimapte v
    float phi = asin(z / *r);
    int64_t v = LookUpV(config, phi * RAD2DEG);

    if (v >= 0) {
        u = (u - config.azimuth_lut_ptr[v] * DEG2RAD) /
            config.azimuth_resolution;
        u = (u < 0) ? u + config.width : u;
        u = (u >= config.width) ? u - config.width : u;
        *ui = static_cast<int64_t>(round(u));
        *vi = static_cast<int64_t>(round(v));
        return true;
    } else {
        *ui = -1;
        *vi = -1;
        return false;
    }
}

inline OPEN3D_DEVICE bool DeviceProjectSimple(
        const LiDARIntrinsicPtrs& config,
        // this transform indexer should be nested:
        // sensor_to_lidar @ rigid_transformation
        const TransformIndexer& transform_indexer,
        float x_in,
        float y_in,
        float z_in,
        int64_t* ui,
        int64_t* vi,
        float* r) {
    float x, y, z;
    transform_indexer.RigidTransform(x_in, y_in, z_in, &x, &y, &z);
    *r = sqrt(x * x + y * y + z * z);

    // Estimate u
    float u = atan2(y, x);
    u = (u < 0) ? TWO_PI + u : u;
    u = (TWO_PI - u) / config.azimuth_resolution;
    u = (u < 0) ? u + config.width : u;
    u = (u >= config.width) ? u - config.width : u;

    // Estimate v
    float phi = asin(z / *r);
    float v = 1 - (phi / DEG2RAD - config.min_altitude) /
                          (config.max_altitude - config.min_altitude);
    v *= config.height;

    if (v >= 0 && v <= config.height - 1) {
        *ui = static_cast<int64_t>(round(u));
        *vi = static_cast<int64_t>(round(v));
        return true;
    } else {
        *ui = -1;
        *vi = -1;
        return false;
    }
}

inline OPEN3D_DEVICE void DeviceUnprojectLUT(const LiDARIntrinsicPtrs& config,
                                             int64_t workload_idx,
                                             float r,
                                             float* x_out,
                                             float* y_out,
                                             float* z_out) {
    int64_t workload_offset = workload_idx * 3;
    *x_out = config.dir_lut_ptr[workload_offset + 0] * r +
             config.offset_lut_ptr[workload_offset + 0];
    *y_out = config.dir_lut_ptr[workload_offset + 1] * r +
             config.offset_lut_ptr[workload_offset + 1];
    *z_out = config.dir_lut_ptr[workload_offset + 2] * r +
             config.offset_lut_ptr[workload_offset + 2];
}

inline OPEN3D_DEVICE void DeviceUnprojectLUT(const LiDARIntrinsicPtrs& config,
                                             int64_t u,
                                             int64_t v,
                                             float r,
                                             float* x_out,
                                             float* y_out,
                                             float* z_out) {
    DeviceUnprojectLUT(config, v * config.width + u, r, x_out, y_out, z_out);
}

inline OPEN3D_DEVICE void DeviceUnprojectSimple(
        const LiDARIntrinsicPtrs& config,
        int64_t u,
        int64_t v,
        float r,
        float* x_out,
        float* y_out,
        float* z_out) {
    float theta = -(2 * float(u) / config.width - 1) * M_PI;

    float phi = float(v) / config.height;
    phi = (1 - phi) * (config.max_altitude - config.min_altitude) +
          config.min_altitude;
    phi = M_PI / 2 - phi * DEG2RAD;

    *x_out = r * sin(phi) * cos(theta);
    *y_out = r * sin(phi) * sin(theta);
    *z_out = r * cos(phi);
}

inline OPEN3D_DEVICE bool GetJacobianPointToPlane(
        const NDArrayIndexer& source_vertex_indexer,
        const NDArrayIndexer& source_mask_indexer,
        const NDArrayIndexer& target_vertex_indexer,
        const NDArrayIndexer& target_mask_indexer,
        const NDArrayIndexer& target_normal_indexer,
        const TransformIndexer& proj_transform,
        const TransformIndexer& src2dst_transform,
        const LiDARIntrinsicPtrs& config,
        float depth_diff,
        int x,
        int y,
        float* J_ij,
        float& r) {
    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    bool mask_v = *source_mask_indexer.GetDataPtr<bool>(x, y);
    if (!mask_v) return false;

    int64_t ui, vi;
    float d;
    bool mask_proj =
            DeviceProjectSimple(config, proj_transform, source_v[0],
                                source_v[1], source_v[2], &ui, &vi, &d);
    if (!mask_proj || !(*target_mask_indexer.GetDataPtr<bool>(ui, vi))) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3];
    src2dst_transform.RigidTransform(
            source_v[0], source_v[1], source_v[2], &T_source_to_target_v[0],
            &T_source_to_target_v[1], &T_source_to_target_v[2]);

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

#ifdef __CUDACC__
void LiDARUnprojectCUDA
#else
void LiDARUnprojectCPU
#endif
        (const core::Tensor& range_image,
         const core::Tensor& transformation,
         const LiDARIntrinsicPtrs& config,
         core::Tensor& xyz_im,
         core::Tensor& mask_im,
         float depth_scale,
         float depth_min,
         float depth_max) {
    core::Device device = range_image.GetDevice();

    core::SizeVector sv = range_image.GetShape();
    int64_t h = sv[0];
    int64_t w = sv[1];
    int64_t n = h * w;

    TransformIndexer transform_indexer(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            transformation.Contiguous());

    const uint16_t* range_image_ptr = range_image.GetDataPtr<uint16_t>();
    float* xyz_im_ptr = xyz_im.GetDataPtr<float>();
    bool* mask_im_ptr = mask_im.GetDataPtr<bool>();

    float depth_max_scaled = depth_max * depth_scale;
    float depth_min_scaled = depth_min * depth_scale;

    if (config.has_lut) {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
            float range = range_image_ptr[workload_idx];
            int u = workload_idx % w;
            int v = workload_idx / w;

            if (range > depth_max_scaled || range <= depth_min_scaled) {
                xyz_im_ptr[workload_idx * 3 + 0] = 0;
                xyz_im_ptr[workload_idx * 3 + 1] = 0;
                xyz_im_ptr[workload_idx * 3 + 2] = 0;
                mask_im_ptr[workload_idx] = false;
                return;
            }
            mask_im_ptr[workload_idx] = true;

            float x, y, z;
            DeviceUnprojectSimple(config, u, v, range / depth_scale, &x, &y,
                                  &z);

            int64_t workload_offset = workload_idx * 3;
            transform_indexer.RigidTransform(x, y, z,
                                             &xyz_im_ptr[workload_offset + 0],
                                             &xyz_im_ptr[workload_offset + 1],
                                             &xyz_im_ptr[workload_offset + 2]);
        });
    } else {
        utility::LogError("Unimplemented without a lut.");
    }
}

#ifdef __CUDACC__
void LiDARProjectCUDA
#else
void LiDARProjectCPU
#endif
        (const core::Tensor& xyz,
         const core::Tensor& transformation,
         const LiDARIntrinsicPtrs& config,
         core::Tensor& us,
         core::Tensor& vs,
         core::Tensor& rs,
         core::Tensor& masks) {
    core::Device device = xyz.GetDevice();
    int64_t n = xyz.GetLength();

    // TODO: make them configurable
    TransformIndexer transform_indexer(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            transformation.Contiguous());

    const float* xyz_ptr = xyz.GetDataPtr<float>();
    int64_t* u_ptr = us.GetDataPtr<int64_t>();
    int64_t* v_ptr = vs.GetDataPtr<int64_t>();
    float* r_ptr = rs.GetDataPtr<float>();
    bool* mask_ptr = masks.GetDataPtr<bool>();

    if (config.has_lut) {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
            int64_t workload_offset = 3 * workload_idx;
            mask_ptr[workload_idx] = DeviceProjectSimple(
                    config, transform_indexer, xyz_ptr[workload_offset + 0],
                    xyz_ptr[workload_offset + 1], xyz_ptr[workload_offset + 2],
                    &u_ptr[workload_idx], &v_ptr[workload_idx],
                    &r_ptr[workload_idx]);
        });
    } else {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
            int64_t workload_offset = 3 * workload_idx;
            mask_ptr[workload_idx] = DeviceProjectSimple(
                    config, transform_indexer, xyz_ptr[workload_offset + 0],
                    xyz_ptr[workload_offset + 1], xyz_ptr[workload_offset + 2],
                    &u_ptr[workload_idx], &v_ptr[workload_idx],
                    &r_ptr[workload_idx]);
        });
    }
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
