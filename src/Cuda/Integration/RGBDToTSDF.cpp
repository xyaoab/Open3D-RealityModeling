#include "RGBDToTSDF.h"
#include <Cuda/Common/JacobianCuda.h>
#include <iostream>

namespace open3d {
namespace cuda {
std::tuple<bool, Eigen::Matrix4d, float> RGBDToTSDFRegistration(
        RGBDImageCuda &rgbd,
        ScalableTSDFVolumeCuda &volume,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world) {
    ArrayCuda<float> linear_system;
    linear_system.Create(29);

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float loss, inliers;

    bool is_success;
    Eigen::Matrix4d cam_to_world = transform_camera_to_world.ToEigen();
    Eigen::Matrix4d delta;

    for (int iter = 0; iter < 3; ++iter) {
        linear_system.Memset(0);
        transform_camera_to_world.FromEigen(cam_to_world);

        BuildLinearSystemRGBDToTSDFKernelCaller(
                rgbd, volume, camera, transform_camera_to_world, linear_system);

        std::vector<float> results = linear_system.DownloadAll();
        ExtractResults(results, JtJ, Jtr, loss, inliers);

        utility::LogInfo("> Iter {}: loss = {}, avg loss = {}, inliers = {}",
                         iter, loss, loss / inliers, inliers);
        std::cout << cam_to_world << "\n";

        std::tie(is_success, delta) =
                utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);
        cam_to_world = delta * cam_to_world;
    }

    return std::make_tuple(is_success, cam_to_world, loss / inliers);
}
}  // namespace cuda
}  // namespace open3d
