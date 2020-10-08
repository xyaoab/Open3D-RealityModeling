//
// Created by wei on 10/10/18.
//

#pragma once

#include "UniformTSDFVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {

__global__ void ResetUniformTSDFVolumeKernel(
        UniformTSDFVolumeCudaDevice server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= server.N_ || y >= server.N_ || z >= server.N_) return;

    Vector3i X = Vector3i(x, y, z);
    server.tsdf(X) = 1.0f;
    server.fg(X) = 1;
    server.bg(X) = 1;
}

__host__ void UniformTSDFVolumeCudaKernelCaller::Reset(
        UniformTSDFVolumeCuda &volume) {
    const int num_blocks = DIV_CEILING(volume.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    ResetUniformTSDFVolumeKernel<<<blocks, threads>>>(*volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void IntegrateKernel(UniformTSDFVolumeCudaDevice server,
                                RGBDImageCudaDevice rgbd,
                                PinholeCameraIntrinsicCuda camera,
                                TransformCuda transform_camera_to_world,
                                ImageCudaDevice<uchar, 1> mask) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= server.N_ || y >= server.N_ || z >= server.N_) return;

    Vector3i X = Vector3i(x, y, z);
    server.Integrate(X, rgbd, camera, transform_camera_to_world, mask);
}

__host__ void UniformTSDFVolumeCudaKernelCaller::Integrate(
        UniformTSDFVolumeCuda &volume,
        RGBDImageCuda &rgbd,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        ImageCuda<uchar, 1>& mask) {
    const int num_blocks = DIV_CEILING(volume.N_, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel<<<blocks, threads>>>(*volume.device_, *rgbd.device_, camera,
                                         transform_camera_to_world, *mask.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void RayCastingKernel(UniformTSDFVolumeCudaDevice server,
                                 ImageCudaDevice<float, 3> vertex,
                                 ImageCudaDevice<float, 3> normal,
                                 ImageCudaDevice<uchar, 3> color,
                                 PinholeCameraIntrinsicCuda camera,
                                 TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= color.width_ || y >= color.height_) return;

    Vector2i p = Vector2i(x, y);
    Vector3f v, n;
    Vector3b c;
    bool mask = server.RayCasting(p, v, n, c, camera, transform_camera_to_world);

    if (!mask) {
        vertex.at(x, y) = Vector3f(nanf("nan"));
        normal.at(x, y) = Vector3f(nanf("nan"));
        color.at(x, y) = Vector3b(0);
        return;
    }
    vertex.at(x, y) = v;
    normal.at(x, y) = n;
    color.at(x, y) = c;

}

void UniformTSDFVolumeCudaKernelCaller::RayCasting(
        UniformTSDFVolumeCuda &volume,
        ImageCuda<float, 3> &vertex,
        ImageCuda<float, 3> &normal,
        ImageCuda<uchar, 3> &color,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    const dim3 blocks(DIV_CEILING(vertex.width_, THREAD_2D_UNIT),
                      DIV_CEILING(vertex.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel<<<blocks, threads>>>(*volume.device_, *vertex.device_, *normal.device_, *color.device_,
                                          camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}  // namespace cuda
}  // namespace open3d
