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

#include <iostream>
#include <memory>

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TSDFBundleAdjustment [options]");
    utility::LogInfo("      Integrate RGBD stream and extract geometry.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --match file              : The match file of an RGBD stream. Must have.");
    utility::LogInfo("    --save_pointcloud         : Save a point cloud created by marching cubes.");
    utility::LogInfo("    --save_mesh               : Save a mesh created by marching cubes.");
    utility::LogInfo("    --save_voxel              : Save a point cloud of the TSDF voxel.");
    utility::LogInfo("    --every_k_frames k        : Save/reset every k frames. Default: 0 (none).");
    utility::LogInfo("    --length l                : Length of the volume, in meters. Default: 4.0.");
    utility::LogInfo("    --resolution r            : Resolution of the voxel grid. Default: 512.");
    utility::LogInfo("    --sdf_trunc_percentage t  : TSDF truncation percentage, of the volume length. Default: 0.01.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }

    std::string match_filename =
            utility::GetProgramOptionAsString(argc, argv, "--match");
    // bool save_pointcloud =
    //         utility::ProgramOptionExists(argc, argv, "--save_pointcloud");
    bool save_mesh = utility::ProgramOptionExists(argc, argv, "--save_mesh");
    // bool save_voxel = utility::ProgramOptionExists(argc, argv,
    // "--save_voxel");
    // int every_k_frames =
    //         utility::GetProgramOptionAsInt(argc, argv, "--every_k_frames",
    //         0);
    double length =
            utility::GetProgramOptionAsDouble(argc, argv, "--length", 4.0);
    int resolution =
            utility::GetProgramOptionAsInt(argc, argv, "--resolution", 512);
    double sdf_trunc_percentage = utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc_percentage", 0.01);
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);

    /// Buffer for IO
    FILE *file = utility::filesystem::FOpen(match_filename, "r");
    if (file == NULL) {
        utility::LogWarning("Unable to open file {}", match_filename);
        fclose(file);
        return 0;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];

    int index = 0;

    /// Shared intrinsics
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    /// Volume for integration
    integration::ScalableTSDFVolume volume(
            length / (double)resolution, length * sdf_trunc_percentage,
            integration::TSDFVolumeColorType::RGB8);

    /// Image buffer
    geometry::Image depth, color;
    std::shared_ptr<geometry::RGBDImage> prev_rgbd = nullptr;

    Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();

    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::vector<std::string> st;
        utility::SplitString(st, buffer, "\t\r\n ");

        if (st.size() >= 2) {
            utility::LogInfo("Processing frame {:d} ...", index);
            io::ReadImage(st[0], depth);
            io::ReadImage(st[1], color);

            auto curr_rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
                    color, depth, 1000.0, 4.0, false);

            Eigen::Matrix4d init_odo = Eigen::Matrix4d::Identity();
            if (prev_rgbd != nullptr) {
                std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
                        odometry::ComputeRGBDOdometry(
                                *curr_rgbd, *prev_rgbd, intrinsic, init_odo,
                                odometry::RGBDOdometryJacobianFromHybridTerm(),
                                odometry::OdometryOption());
                extrinsics = std::get<1>(rgbd_odo) * extrinsics;
            }

            volume.Integrate(*curr_rgbd, intrinsic, extrinsics.inverse());
            prev_rgbd = curr_rgbd;

            index++;
        }
    }
    fclose(file);

    if (save_mesh) {
        utility::LogInfo("Saving mesh ...");
        auto mesh = volume.ExtractTriangleMesh();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("integrated.ply", *mesh);
    }

    return 0;
}
