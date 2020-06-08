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

    __shared__ float local_sum0[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum1[THREAD_2D_UNIT * THREAD_2D_UNIT];
    __shared__ float local_sum2[THREAD_2D_UNIT * THREAD_2D_UNIT];
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    // Boundary check
    if (x >= rgbd.width_ || y >= rgbd.height_) return;

    // Depth check
    float d = rgbd.depth_.at(x, y)(0);
    if (d <= 0.0 || d >= 3.0f) return;

    Vector3f Xw =
            T_cam_to_world * intrinsic.InverseProjectPixel(Vector2i(x, y), d);
    Vector3f X = volume.world_to_voxelf(Xw);

    // Voxel check
    float weight = volume.WeightAt(X);
    bool mask = (weight > 0);

    // Truncation check
    float tsdf = volume.TSDFAt(X);
    mask = mask && (-1.0f <= tsdf && tsdf <= 1.0f);

    // Build linear system
    Vector6f jacobian;
    float residual = 0;
    if (mask) {
        Vector3f tsdf_grad = volume.GradientAt(X) / volume.voxel_length_;
        jacobian(0) = -Xw(2) * tsdf_grad(1) + Xw(1) * tsdf_grad(2);
        jacobian(1) = Xw(2) * tsdf_grad(0) - Xw(0) * tsdf_grad(2);
        jacobian(2) = -Xw(1) * tsdf_grad(0) + Xw(0) * tsdf_grad(1);
        jacobian(3) = tsdf_grad(0);
        jacobian(4) = tsdf_grad(1);
        jacobian(5) = tsdf_grad(2);
        residual = tsdf;
    }

    HessianCuda<6> JtJ;
    Vector6f Jtr;
    ComputeJtJAndJtr(jacobian, residual, JtJ, Jtr);

    /** Reduce Sum JtJ -> 2ms **/
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = mask ? JtJ(i + 0) : 0;
        local_sum1[tid] = mask ? JtJ(i + 1) : 0;
        local_sum2[tid] = mask ? JtJ(i + 2) : 0;
        __syncthreads();

        BlockReduceSum<float, THREAD_2D_UNIT * THREAD_2D_UNIT>(
                tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&linear_system.at(i + 0), local_sum0[0]);
            atomicAdd(&linear_system.at(i + 1), local_sum1[0]);
            atomicAdd(&linear_system.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum Jtr **/
    const int OFFSET1 = 21;
    for (size_t i = 0; i < 6; i += 3) {
        local_sum0[tid] = mask ? Jtr(i + 0) : 0;
        local_sum1[tid] = mask ? Jtr(i + 1) : 0;
        local_sum2[tid] = mask ? Jtr(i + 2) : 0;
        __syncthreads();

        BlockReduceSum<float, THREAD_2D_UNIT * THREAD_2D_UNIT>(
                tid, local_sum0, local_sum1, local_sum2);

        if (tid == 0) {
            atomicAdd(&linear_system.at(i + 0 + OFFSET1), local_sum0[0]);
            atomicAdd(&linear_system.at(i + 1 + OFFSET1), local_sum1[0]);
            atomicAdd(&linear_system.at(i + 2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum loss and inlier **/
    const int OFFSET2 = 27;
    {
        local_sum0[tid] = mask ? residual : 0;
        local_sum1[tid] = mask ? 1 : 0;
        __syncthreads();

        BlockReduceSum<float, THREAD_2D_UNIT * THREAD_2D_UNIT>(tid, local_sum0,
                                                               local_sum1);

        if (tid == 0) {
            atomicAdd(&linear_system.at(0 + OFFSET2), local_sum0[0]);
            atomicAdd(&linear_system.at(1 + OFFSET2), local_sum1[0]);
        }
        __syncthreads();
    }
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
