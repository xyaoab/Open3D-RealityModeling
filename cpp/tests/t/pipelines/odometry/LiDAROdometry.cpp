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
    t::pipelines::odometry::LiDARIntrinsic calib(calib_npz, device);

    t::geometry::LiDARImage src =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000000.png"))
                    ->To(device);

    core::Tensor xyz_im, mask_im;
    core::Tensor transformation =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());

    auto vis = src.Visualize(calib);
    auto vis_ptr = std::make_shared<open3d::geometry::Image>(
            vis.ColorizeDepth(1000.0, 0.0, 30.0).ToLegacy());
    visualization::DrawGeometries({vis_ptr});

    std::tie(xyz_im, mask_im) =
            src.Unproject(calib, transformation, 0.0, 100.0);

    auto pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud(xyz_im.IndexGet({mask_im})).ToLegacy());
    visualization::DrawGeometries({pcd_ptr});
}

TEST_P(LiDAROdometryPermuteDevices, Project) {
    core::Device device = GetParam();

    std::string calib_npz =
            utility::GetDataPathCommon("LiDARICP/ouster_calib.npz");
    t::pipelines::odometry::LiDARIntrinsic calib(calib_npz, device);

    t::geometry::LiDARImage src =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000000.png"))
                    ->To(device);
    auto raw_im_ptr = std::make_shared<open3d::geometry::Image>(
            src.ColorizeDepth(1000.0, 0.0, 30.0).ToLegacy());
    visualization::DrawGeometries({raw_im_ptr});

    core::Tensor xyz_im, mask_im;
    core::Tensor transformation =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());
    std::tie(xyz_im, mask_im) =
            src.Unproject(calib, transformation, 0.0, 100.0);

    /// (N, 3)
    core::Tensor xyz = xyz_im.IndexGet({mask_im});

    core::Tensor u, v, r, mask;
    std::tie(u, v, r, mask) = src.Project(xyz, calib);

    core::Tensor rendered = core::Tensor::Zeros(
            core::SizeVector{128, 1024}, core::Dtype::Float32, src.GetDevice());
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
    utility::LogInfo("pcd_ptr points: {}", pcd_ptr->points_.size());
    visualization::DrawGeometries({pcd_ptr});
}

void VisualizeRegistration(const t::geometry::LiDARImage &src,
                           const t::geometry::LiDARImage &dst,
                           const core::Tensor &transformation,
                           const t::pipelines::odometry::LiDARIntrinsic &calib,
                           float depth_min,
                           float depth_max) {
    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());

    core::Tensor src_xyz_map, src_mask_map, dst_xyz_map, dst_mask_map;

    std::tie(src_xyz_map, src_mask_map) =
            src.Unproject(calib, transformation, depth_min, depth_max);
    std::tie(dst_xyz_map, dst_mask_map) =
            dst.Unproject(calib, identity, depth_min, depth_max);

    t::geometry::PointCloud src_pcd(src_xyz_map.IndexGet({src_mask_map}));
    auto src_pcd_l =
            std::make_shared<open3d::geometry::PointCloud>(src_pcd.ToLegacy());
    src_pcd_l->PaintUniformColor({1, 0, 0});

    t::geometry::PointCloud dst_pcd(dst_xyz_map.IndexGet({dst_mask_map}));
    auto dst_pcd_l =
            std::make_shared<open3d::geometry::PointCloud>(dst_pcd.ToLegacy());
    dst_pcd_l->PaintUniformColor({0, 1, 0});

    visualization::DrawGeometries({src_pcd_l, dst_pcd_l});
}

TEST_P(LiDAROdometryPermuteDevices, Odometry) {
    core::Device device = GetParam();

    const float depth_min = 0.0;
    const float depth_max = 100.0;
    const float depth_diff = 0.3;

    const t::pipelines::odometry::OdometryConvergenceCriteria criteria(20, 1e-6,
                                                                       1e-6);

    std::string calib_npz =
            utility::GetDataPathCommon("LiDARICP/ouster_calib.npz");
    t::pipelines::odometry::LiDARIntrinsic calib(calib_npz, device);

    t::geometry::LiDARImage src =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000000.png"))
                    ->To(device);
    t::geometry::LiDARImage dst =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000010.png"))
                    ->To(device);

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());

    VisualizeRegistration(src, dst, identity, calib, depth_min, depth_max);
    auto result = LiDAROdometry(src, dst, calib, identity, depth_min, depth_max,
                                depth_diff, criteria);
    VisualizeRegistration(src, dst, result.transformation_, calib, depth_min,
                          depth_max);
}

TEST_P(LiDAROdometryPermuteDevices, OdometryNormalDecoupled) {
    core::Device device = GetParam();

    const float depth_min = 0.0;
    const float depth_max = 100.0;
    const float depth_diff = 0.3;

    const t::pipelines::odometry::OdometryConvergenceCriteria criteria(20, 1e-6,
                                                                       1e-6);

    std::string calib_npz =
            utility::GetDataPathCommon("LiDARICP/ouster_calib.npz");
    t::pipelines::odometry::LiDARIntrinsic calib(calib_npz, device);

    t::geometry::LiDARImage src =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000000.png"))
                    ->To(device);
    t::geometry::LiDARImage dst =
            t::io::CreateImageFromFile(
                    utility::GetDataPathCommon("LiDARICP/outdoor/000010.png"))
                    ->To(device);

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());

    core::Tensor dst_normal_map = dst.GetNormalMap(calib);
    VisualizeRegistration(src, dst, identity, calib, depth_min, depth_max);
    auto result = LiDAROdometry(src, dst, dst_normal_map, calib, identity,
                                depth_min, depth_max, depth_diff, criteria);
    VisualizeRegistration(src, dst, result.transformation_, calib, depth_min,
                          depth_max);
}
}  // namespace tests
}  // namespace open3d
