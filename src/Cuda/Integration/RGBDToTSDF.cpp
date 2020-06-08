#include "RGBDToTSDF.h"
#include <Cuda/Common/JacobianCuda.h>

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
    Eigen::Matrix4d extrinsic;

    for (int iter = 0; iter < 10; ++iter) {
        linear_system.Memset(0);
        BuildLinearSystemRGBDToTSDFKernelCaller(
                rgbd, volume, camera, transform_camera_to_world, linear_system);

        std::vector<float> results = linear_system.DownloadAll();
        ExtractResults(results, JtJ, Jtr, loss, inliers);
        utility::LogInfo("> Iter {}: loss = {}, avg loss = {}, inliers = {}",
                         iter, loss, loss / inliers, inliers);

        std::tie(is_success, extrinsic) =
                utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

        transform_camera_to_world.FromEigen(extrinsic);
    }

    return std::make_tuple(is_success, extrinsic, loss / inliers);
}
}  // namespace cuda
}  // namespace open3d
