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

#include "open3d/t/pipelines/odometry/LiDAROdometry.h"

#include "core/CoreTest.h"
#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class LiDAROdometryPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(LiDAROdometry,
                         LiDAROdometryPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(LiDAROdometryPermuteDevices, Unproject) {
    core::Device device = GetParam();

    std::string calib_npz =
            utility::GetDataPathCommon("LiDARICP/ouster_calib.npz");
    t::pipelines::odometry::LiDARCalib calib(calib_npz, device);

    t::geometry::Image src_depth = *t::io::CreateImageFromFile(
            utility::GetDataPathCommon("LiDARICP/000000.png"));

    core::Tensor xyz_im, mask_im;
    core::Tensor depth = src_depth.AsTensor().To(device);
    std::tie(xyz_im, mask_im) = calib.Unproject(depth, 0.0, 100.0);

    auto pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud(xyz_im.IndexGet({mask_im})).ToLegacy());
    visualization::DrawGeometries({pcd_ptr});
}

TEST_P(LiDAROdometryPermuteDevices, Project) {
    core::Device device = GetParam();

    std::string calib_npz =
            utility::GetDataPathCommon("LiDARICP/ouster_calib.npz");
    t::pipelines::odometry::LiDARCalib calib(calib_npz, device);

    t::geometry::Image src_depth = *t::io::CreateImageFromFile(
            utility::GetDataPathCommon("LiDARICP/000000.png"));
    auto raw_im_ptr = std::make_shared<open3d::geometry::Image>(
            src_depth.ColorizeDepth(1000.0, 0.0, 30.0).ToLegacy());
    visualization::DrawGeometries({raw_im_ptr});

    core::Tensor xyz_im, mask_im;
    core::Tensor depth = src_depth.AsTensor().To(device);

    std::tie(xyz_im, mask_im) = calib.Unproject(depth, 0.0, 100.0);

    /// (N, 3)
    core::Tensor xyz = xyz_im.IndexGet({mask_im});

    core::Tensor u, v, r, mask;
    std::tie(u, v, r, mask) = calib.Project(xyz);

    core::Tensor rendered =
            core::Tensor::Zeros(core::SizeVector{128, 1024},
                                core::Dtype::Float32, depth.GetDevice());
    auto u_mask = u.IndexGet({mask});
    auto v_mask = v.IndexGet({mask});
    auto r_mask = r.IndexGet({mask});
    rendered.IndexSet({v_mask, u_mask}, r_mask);

    t::geometry::Image rendered_im(rendered);
    auto colorized_im_ptr = std::make_shared<open3d::geometry::Image>(
            rendered_im.ColorizeDepth(1.0, 0.0, 30.0).ToLegacy());
    visualization::DrawGeometries({colorized_im_ptr});

    auto pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud(xyz_im.IndexGet({v_mask, u_mask}))
                    .ToLegacy());
    visualization::DrawGeometries({pcd_ptr});
}

TEST_P(LiDAROdometryPermuteDevices, ComputeOdometryResultPointToPlane) {
    core::Device device = GetParam();

    // const float depth_scale = 1000.0;
    // const float depth_diff = 0.2;

    t::geometry::Image src_depth = *t::io::CreateImageFromFile(
            utility::GetDataPathCommon("LiDARICP/000000.png"));
    t::geometry::Image dst_depth = *t::io::CreateImageFromFile(
            utility::GetDataPathCommon("LiDARICP/000010.png"));

    src_depth = src_depth.To(device);
    dst_depth = dst_depth.To(device);
}
}  // namespace tests
}  // namespace open3d
