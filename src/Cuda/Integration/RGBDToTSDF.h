#pragma once

#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

namespace open3d {
namespace cuda {

void BuildLinearSystemRGBDToTSDFKernelCaller(
        RGBDImageCuda &rgbd,
        ScalableTSDFVolumeCuda &volume,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world,
        ArrayCuda<float> &linear_system);

std::tuple<bool, Eigen::Matrix4d, float> RGBDToTSDFRegistration(
        RGBDImageCuda &rgbd,
        ScalableTSDFVolumeCuda &volume,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world);
}  // namespace cuda
}  // namespace open3d
