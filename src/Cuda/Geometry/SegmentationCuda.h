//
// Created by akash on 2020-05-20
//

#pragma once

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Geometry/ImageCuda.h>
#include <Eigen/Eigen>

namespace open3d {
namespace cuda {

class SegmentationCuda {
public:
    static ImageCuda<float, 1> ComputeEdgeMap(ImageCuda<float, 3> &vertex_map,
                               ImageCuda<float, 3> &normal_map);
};

class SegmentationKernelCaller {
public:
    static void ComputeEdgeMap(ImageCuda<float, 3> &vertex_map,
                        ImageCuda<float, 3> &normal_map,
                        ImageCuda<float, 1> &dst_edge_map);
};
__DEVICE__
float GetConcaveTermKernel(Vector3f vertex_i, Vector3f normal_i,
                           Vector3f vertex, Vector3f normal);

__DEVICE__
float GetDiscontinuityTermKernel(Vector3f vertex_i, Vector3f vertex,
                                 Vector3f normal);

__GLOBAL__
void computeEdgeMapKernel(ImageCudaDevice<float, 3> vertex_map,
                          ImageCudaDevice<float, 3> normal_map,
                          ImageCudaDevice<float, 1> dst_edge_map);

} //namespace cuda
} //namespace open3d
