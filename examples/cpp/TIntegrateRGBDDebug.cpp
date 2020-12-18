// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <fstream>
#include <iomanip>

#include "open3d/Open3D.h"
using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    TIntegrateRGBD [color_folder] [depth_folder] [trajectory] [options]");
    utility::LogInfo("     Given RGBD images, reconstruct mesh or point cloud from color and depth images");
    utility::LogInfo("     [options]");
    utility::LogInfo("     --camera_intrinsic [intrinsic_path]");
    utility::LogInfo("     --device [CPU:0]");
    utility::LogInfo("     --mesh");
    utility::LogInfo("     --pointcloud");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char** argv) {
    std::string data_path = "/home/wei/Workspace/data/bundlefusion/copyroom/";
    camera::PinholeCameraIntrinsic intrinsic;
    intrinsic.SetIntrinsics(640, 480, 583, 583, 320, 240);
    std::string device_code = "cuda:0";
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, Dtype::Float32);

    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Dtype::Float32},
                                           {"weight", core::Dtype::UInt16},
                                           {"color", core::Dtype::UInt16}},
                                          3.0f / 512.f, 0.04f, 16, 100, device);
    for (int frame_idx = 0; frame_idx < 4478; frame_idx++) {
        // cv::Mat ray_cast;
        std::cout << frame_idx << std::endl;
        std::ostringstream curr_frame_prefix;
        curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
        std::string cam2world_file =
                data_path + "frame-" + curr_frame_prefix.str() + ".pose.txt";
        std::string depth_im_file =
                data_path + "frame-" + curr_frame_prefix.str() + ".depth.png";
        std::string rgb_im_file =
                data_path + "frame-" + curr_frame_prefix.str() + ".color.jpg";
        std::shared_ptr<geometry::Image> depth_legacy =
                io::CreateImageFromFile(depth_im_file);
        std::shared_ptr<geometry::Image> color_legacy =
                io::CreateImageFromFile(rgb_im_file);
        t::geometry::Image depth =
                t::geometry::Image::FromLegacyImage(*depth_legacy, device);
        t::geometry::Image color =
                t::geometry::Image::FromLegacyImage(*color_legacy, device);
        std::ifstream readin(cam2world_file, std::ios::in);
        if (readin.fail() || readin.eof()) {
            std::cout << "ERROR: Can not read pose file " << cam2world_file
                      << std::endl;
            return false;
        }
        Eigen::Matrix4f cam2world;

        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) {
                readin >> cam2world(r, c);
            }

        Eigen::Matrix4f extrinsic = cam2world.inverse();
        Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic).Copy(
                        device);
        utility::Timer timer;
        timer.Start();
        voxel_grid.Integrate(depth, color, intrinsic_t, extrinsic_t);
        timer.Stop();
        utility::LogInfo("{}: Integration takes {}", frame_idx,
                         timer.GetDuration());
    }

    auto mesh = voxel_grid.ExtractSurfaceMesh();
    auto mesh_legacy = std::make_shared<geometry::TriangleMesh>(
            mesh.ToLegacyTriangleMesh());
    open3d::io::WriteTriangleMesh("mesh_" + device.ToString() + ".ply",
                                  *mesh_legacy);
}
