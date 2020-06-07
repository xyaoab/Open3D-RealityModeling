#include "RGBDToTSDF.h"

namespace open3d {
namespace cuda {
void RGBDToTSDFRegistration(RGBDImageCuda &rgbd,
                            ScalableTSDFVolumeCuda &volume,
                            PinholeCameraIntrinsicCuda camera,
                            TransformCuda transform_camera_to_world) {
    ArrayCuda<float> linear_system;
    BuildLinearSystemRGBDToTSDFKernelCaller(
            rgbd, volume, camera, transform_camera_to_world, linear_system);
}
}  // namespace cuda
}  // namespace open3d
