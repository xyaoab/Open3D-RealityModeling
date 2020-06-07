#include <Cuda/Geometry/RGBDImageCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include "RGBDToTSDF.h"
#include "ScalableTSDFVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {
__global__ void BuildLinearSystemRGBDToTSDFKernel(
        RGBDImageCudaDevice rgbd,
        ScalableTSDFVolumeCudaDevice volume,
        PinholeCameraIntrinsicCuda intrinsic,
        TransformCuda T_cam_to_world,
        /* 6 x 6 linear system */
        ArrayCudaDevice<float> linear_system) {
    /// \nabla D[X(xi)] = dD/dX dX/dxi
    /// Unproject each pixel into 3D space with \rgbd and \camera,
    /// \transform_camera_to_world
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Boundary check
    if (x >= rgbd.width_ || y >= rgbd.height_) return;

    // Depth check
    float d = rgbd.depth_.at(x, y)(0);
    if (d <= 0.0 || d >= 3.0f) return;

    Vector3f X =
            T_cam_to_world * intrinsic.InverseProjectPixel(Vector2i(x, y), d);

    // Voxel check
    float weight = volume.WeightAt(X);
    if (weight == 0) return;

    // Truncation check
    float tsdf = volume.TSDFAt(X);
    if (tsdf <= -1.0f || tsdf >= 1.0f) return;

    // Build linear system
    Vector3f tsdf_grad = volume.GradientAt(X);

    Vector6f jacobian;
    jacobian(0) = -X(2) * tsdf_grad(1) + X(1) * tsdf_grad(2);
    jacobian(1) = X(2) * tsdf_grad(0) - X(0) * tsdf_grad(2);
    jacobian(2) = -X(1) * tsdf_grad(0) + X(0) * tsdf_grad(1);
    jacobian(3) = tsdf_grad(0);
    jacobian(4) = tsdf_grad(1);
    jacobian(5) = tsdf_grad(2);
}

void BuildLinearSystemRGBDToTSDFKernelCaller(
        RGBDImageCuda &rgbd,
        ScalableTSDFVolumeCuda &volume,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world,
        ArrayCuda<float> &linear_system) {
    const dim3 blocks(DIV_CEILING(rgbd.width_, THREAD_2D_UNIT),
                      DIV_CEILING(rgbd.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BuildLinearSystemRGBDToTSDFKernel<<<blocks, threads>>>(
            *rgbd.device_, *volume.device_, camera, transform_camera_to_world,
            *linear_system.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}  // namespace cuda
}  // namespace open3d