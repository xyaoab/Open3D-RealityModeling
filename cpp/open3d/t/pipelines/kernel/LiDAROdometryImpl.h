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

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

#define PI 3.14159265358979323846
#define TWO_PI (2 * PI)
#define RAD2DEG (180 / PI)
#define DEG2RAD (PI / 180)

#ifdef __CUDACC__
void LiDARUnprojectCUDA
#else
void LiDARUnprojectCPU
#endif
        (const core::Tensor& range_image,
         const core::Tensor& dir_lut,
         const core::Tensor& offset_lut,
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

    const uint16_t* range_image_ptr = range_image.GetDataPtr<uint16_t>();
    const float* dir_lut_ptr = dir_lut.GetDataPtr<float>();
    const float* offset_lut_ptr = offset_lut.GetDataPtr<float>();

    float* xyz_im_ptr = xyz_im.GetDataPtr<float>();
    bool* mask_im_ptr = mask_im.GetDataPtr<bool>();

    float depth_max_scaled = depth_max * depth_scale;
    float depth_min_scaled = depth_min * depth_scale;
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
        float range = range_image_ptr[workload_idx];
        if (range > depth_max_scaled || range < depth_min_scaled) {
            mask_im_ptr[workload_idx] = false;
            return;
        }

        mask_im_ptr[workload_idx] = true;
        int64_t workload_offset = workload_idx * 3;
        for (int i = 0; i < 3; ++i) {
            xyz_im_ptr[workload_offset + i] =
                    dir_lut_ptr[workload_offset + i] * range +
                    offset_lut_ptr[workload_offset + i];
        }
    });
}

#ifdef __CUDACC__
void LiDARProjectCUDA
#else
using std::abs;
using std::asin;
using std::atan2;
using std::max;
using std::min;
using std::round;
using std::sqrt;
void LiDARProjectCPU
#endif
        (const core::Tensor& xyz,
         const core::Tensor& transformation,
         const core::Tensor& azimuth_lut,
         const core::Tensor& altitude_lut,
         const core::Tensor& inv_altitude_lut,
         core::Tensor& us,
         core::Tensor& vs,
         core::Tensor& rs,
         core::Tensor& masks) {
    core::Device device = xyz.GetDevice();
    int64_t n = xyz.GetLength();

    // TODO: make them configurable
    const int64_t width = 1024;
    const int64_t height = altitude_lut.GetLength();
    const int64_t inv_lut_len = inv_altitude_lut.GetLength();

    const float azimuth_resolution = TWO_PI / width;
    const float azimuth_deg_to_pixel = width / 360.0;

    const float altitude_resolution = 0.4;
    const float altitude_min = altitude_lut[height - 1].Item<float>();

    t::geometry::kernel::TransformIndexer transform_indexer(
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device()),
            transformation.Contiguous());

    const float* xyz_ptr = xyz.GetDataPtr<float>();
    int64_t* u_ptr = us.GetDataPtr<int64_t>();
    int64_t* v_ptr = vs.GetDataPtr<int64_t>();
    float* r_ptr = rs.GetDataPtr<float>();
    bool* mask_ptr = masks.GetDataPtr<bool>();

    const float* azimuth_lut_ptr = azimuth_lut.GetDataPtr<float>();
    const float* altitude_lut_ptr = altitude_lut.GetDataPtr<float>();
    const int64_t* inv_altitude_lut_ptr =
            inv_altitude_lut.GetDataPtr<int64_t>();

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
        auto LookUpV = [=] OPEN3D_DEVICE(float phi_deg) -> int64_t {
            int64_t phi_int = static_cast<int64_t>(
                    round((phi_deg - altitude_min) / altitude_resolution));
            if (phi_int < 0 || phi_int >= inv_lut_len) {
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

        int64_t workload_offset = 3 * workload_idx;

        float x, y, z;
        transform_indexer.RigidTransform(
                xyz_ptr[workload_offset + 0], xyz_ptr[workload_offset + 1],
                xyz_ptr[workload_offset + 2], &x, &y, &z);
        float r = sqrt(x * x + y * y + z * z);
        r_ptr[workload_idx] = r;

        // Estimate u
        float u = atan2(y, x);
        u = (u < 0) ? TWO_PI + u : u;
        u = (TWO_PI - u) / azimuth_resolution;

        // Estimate v
        float phi = asin(z / r);
        int64_t v = LookUpV(phi * RAD2DEG);

        if (v >= 0) {
            u -= azimuth_lut_ptr[v] * azimuth_deg_to_pixel;
            u = (u < 0) ? u + width : u;
            u = (u >= width) ? u - width : u;
            u_ptr[workload_idx] = static_cast<int64_t>(u);
            v_ptr[workload_idx] = static_cast<int64_t>(v);
            mask_ptr[workload_idx] = true;
        } else {
            u_ptr[workload_idx] = u;
            v_ptr[workload_idx] = -1;
            mask_ptr[workload_idx] = false;
        }
    });
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
