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

#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

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
}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
