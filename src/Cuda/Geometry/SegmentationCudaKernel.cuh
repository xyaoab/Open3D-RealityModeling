//
// Created by Akash on 2020-05-20
//

#pragma once

#include "SegmentationCuda.h"

namespace open3d {
namespace cuda {

__device__
float GetConcaveTermKernel(Vector3f vertex_i, Vector3f normal_i, Vector3f vertex, Vector3f normal) {
    Vector3f diff = vertex_i - vertex;
    if(normal.dot(diff) > 0) return 0;
    return 1 - normal_i.dot(normal);
}

__device__
float GetDiscontinuityTermKernel(Vector3f vertex_i, Vector3f vertex, Vector3f normal) {
    Vector3f diff = vertex_i - vertex;
    return fabs(normal.dot(diff));
}

__global__
void computeEdgeMapKernel(ImageCudaDevice<float, 3> vertex_map,
                          ImageCudaDevice<float, 3> normal_map,
                          ImageCudaDevice<float, 1> dst_edge_map) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= vertex_map.width_ || v >= vertex_map.height_)
        return;

    //! Edge pixels
    if(u < 1 || v < 1 || u >= vertex_map.width_ - 1 || v >= vertex_map.height_ - 1)
    {
        dst_edge_map.at(u, v) = Vector1f(1.0f);
        return;
    }

    if(vertex_map.at(u, v).IsNaN() || normal_map.at(u, v).IsNaN())
    {
        dst_edge_map.at(u, v) = Vector1f(1.0f);
        return;
    }


    Vector3f vertex = vertex_map.at(u, v);
    Vector3f normal = normal_map.at(u, v);
    if(vertex(2) <= 0.0f)
    {
        dst_edge_map.at(u, v) = Vector1f(1.0f);
        return;
    }

    float concavity = 0.0f;
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u-1, v-1), normal_map.at(u-1, v-1), vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u-1, v),   normal_map.at(u-1, v),   vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u-1, v+1), normal_map.at(u-1, v+1), vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u, v-1),   normal_map.at(u, v-1),   vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u, v+1),   normal_map.at(u, v+1),   vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u+1, v-1), normal_map.at(u+1, v-1), vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u+1, v),   normal_map.at(u+1, v),   vertex, normal), concavity);
    concavity = fmax(GetConcaveTermKernel(vertex_map.at(u+1, v+1), normal_map.at(u+1, v+1), vertex, normal), concavity);

    float discontinuity = 0.0f;
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v-1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u-1, v+1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u, v-1),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u, v+1),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v-1), vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v),   vertex, normal), discontinuity);
    discontinuity = fmax(GetDiscontinuityTermKernel(vertex_map.at(u+1, v+1), vertex, normal), discontinuity);

    float edge = max(concavity, discontinuity);
    dst_edge_map.at(u, v) = Vector1f(fmin(1.0f, edge));
}

__host__
void SegmentationKernelCaller::ComputeEdgeMap(
        ImageCuda<float, 3> &vertex_map,
        ImageCuda<float, 3> &normal_map,
        ImageCuda<float, 1> &dst_edge_map) {

    const dim3 blocks(DIV_CEILING(vertex_map.width_, THREAD_2D_UNIT),
            DIV_CEILING(vertex_map.width_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    computeEdgeMapKernel<<<blocks, threads>>>(*vertex_map.device_, *normal_map.device_, *dst_edge_map.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


} //namespace cuda
} //namespace open3d
