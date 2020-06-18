//
// Created by Akash on 2020-05-20
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include "SegmentationCuda.h"

namespace open3d {
namespace cuda {

__device__
float GetConcaveTermKernel(Vector3f vertex_i, Vector3f normal_i, Vector3f vertex, Vector3f normal) {
    Vector3f diff = vertex - vertex_i;
    if(diff.dot(normal) > 0) return 1;
    return normal_i.dot(normal);
}

__device__
float GetDiscontinuityTermKernel(Vector3f vertex_i, Vector3f vertex, Vector3f normal) {
    Vector3f diff = vertex_i - vertex;
    return fabs(diff.dot(normal));
}

__global__
void computeEdgeMapKernel(ImageCudaDevice<float, 3> vertex_map,
                          ImageCudaDevice<float, 3> normal_map,
                          ImageCudaDevice<uchar, 1> dst_edge_map) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= vertex_map.width_ || v >= vertex_map.height_)
        return;

    //! Edge pixels
    if(u < 1 || v < 1 || u >= vertex_map.width_ - 1 || v >= vertex_map.height_ - 1)
    {
        dst_edge_map.at(u, v) = Vector1b(0);
        return;
    }

    if(vertex_map.at(u, v).IsNaN() || normal_map.at(u, v).IsNaN())
    {
        dst_edge_map.at(u, v) = Vector1b(0);
        return;
    }


    Vector3f vertex = vertex_map.at(u, v);
    Vector3f normal = normal_map.at(u, v);
    if(vertex(2) <= 0.0f)
    {
        dst_edge_map.at(u, v) = Vector1b(0);
        return;
    }

    float concavity = 1.0f;
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u-1, v-1), normal_map.at(u-1, v-1), vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u-1, v),   normal_map.at(u-1, v),   vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u-1, v+1), normal_map.at(u-1, v+1), vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u, v-1),   normal_map.at(u, v-1),   vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u, v+1),   normal_map.at(u, v+1),   vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u+1, v-1), normal_map.at(u+1, v-1), vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u+1, v),   normal_map.at(u+1, v),   vertex, normal), concavity);
    concavity = fmin(GetConcaveTermKernel(vertex_map.at(u+1, v+1), normal_map.at(u+1, v+1), vertex, normal), concavity);
    concavity = (concavity <= 0.96)? 0 : 255;

    float discontinuity = 0.0f;
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v-1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v+1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u, v-1),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u, v+1),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v-1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v+1), vertex, normal), discontinuity);
    discontinuity = (discontinuity >= 0.001)? 0 : 255;

    float edge = min(concavity, discontinuity);
    dst_edge_map.at(u, v) = Vector1b(edge);
}

__host__
void SegmentationKernelCaller::ComputeEdgeMap(
        ImageCuda<float, 3> &vertex_map,
        ImageCuda<float, 3> &normal_map,
        ImageCuda<uchar, 1> &dst_edge_map) {

    const dim3 blocks(DIV_CEILING(vertex_map.width_, THREAD_2D_UNIT),
            DIV_CEILING(vertex_map.width_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    computeEdgeMapKernel<<<blocks, threads>>>(*vertex_map.device_, *normal_map.device_, *dst_edge_map.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void TransformAndProjectKernel(ImageCudaDevice<uchar, 1> src_mask,
                                ImageCudaDevice<float, 1> src_depth,
                                ImageCudaDevice<uchar, 1> target_mask,
                                TransformCuda transform_source_to_target,
                                PinholeCameraIntrinsicCuda intrinsic)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= src_mask.width_ || v >= src_mask.height_)
        return;

    if (src_mask.at(u, v) == Vector1b(0))
        return;

    float depth = src_depth.at(u, v, 0);
    if (isnan(depth) || depth < 0 || depth > 3.0)
        return;

    Vector3f X_source_on_target = transform_source_to_target * intrinsic.InverseProjectPixel(Vector2i(u, v), depth);

    Vector2f p_targetf = intrinsic.ProjectPoint(X_source_on_target);
    Vector2i p_targeti = Vector2i(int(p_targetf(0)), int(p_targetf(1)));

    if(p_targeti(0) > 0 && p_targeti(0) < src_mask.width_ && p_targeti(1) > 0 && p_targeti(1) < src_mask.height_)
    {
        target_mask.at(p_targeti(0), p_targeti(1)) = Vector1b(255);
    }
}

__host__
void SegmentationKernelCaller::TransformAndProject(
        ImageCuda<uchar, 1> &src_mask,
        ImageCuda<float, 1> &src_depth,
        ImageCuda<uchar, 1> &target_mask,
        TransformCuda &transform_source_to_target,
        PinholeCameraIntrinsicCuda &intrinsic)
{
    const dim3 blocks(DIV_CEILING(src_mask.width_, THREAD_2D_UNIT),
            DIV_CEILING(src_mask.width_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    TransformAndProjectKernel<<<blocks, threads>>>(*src_mask.device_, *src_depth.device_, *target_mask.device_, transform_source_to_target, intrinsic);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


} //namespace cuda
} //namespace open3d
