//
// Created by akash on 2020-05-20
//

#pragma once

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Geometry/ImageCuda.h>

#include <Eigen/Eigen>

#include "Cuda/Common/TransformCuda.h"

namespace open3d {
namespace cuda {

class SegmentationCuda {
public:
    static ImageCuda<uchar, 1> ComputeEdgeMap(ImageCuda<float, 3> &vertex_map,
                                              ImageCuda<float, 3> &normal_map);

    static ImageCuda<uchar, 1> TransformAndProject(
            ImageCuda<uchar, 1> &src_mask,
            ImageCuda<float, 1> &src_depth,
            TransformCuda &transform_source_to_target,
            PinholeCameraIntrinsicCuda &intrinsic);
};

class SegmentationKernelCaller {
public:
    static void ComputeEdgeMap(ImageCuda<float, 3> &vertex_map,
                               ImageCuda<float, 3> &normal_map,
                               ImageCuda<uchar, 1> &dst_edge_map);

    static void TransformAndProject(ImageCuda<uchar, 1> &src_mask,
                                    ImageCuda<float, 1> &src_depth,
                                    ImageCuda<uchar, 1> &target_mask,
                                    TransformCuda &transform_source_to_target,
                                    PinholeCameraIntrinsicCuda &intrinsic);
};
__DEVICE__
float GetConcaveTermKernel(Vector3f vertex_i,
                           Vector3f normal_i,
                           Vector3f vertex,
                           Vector3f normal);

__DEVICE__
float GetDiscontinuityTermKernel(Vector3f vertex_i,
                                 Vector3f vertex,
                                 Vector3f normal);

__GLOBAL__
void computeEdgeMapKernel(ImageCudaDevice<float, 3> vertex_map,
                          ImageCudaDevice<float, 3> normal_map,
                          ImageCudaDevice<uchar, 1> dst_edge_map);

__GLOBAL__
void TransformAndProjectKernel(ImageCudaDevice<uchar, 1> src_mask,
                               ImageCudaDevice<float, 1> src_depth,
                               ImageCudaDevice<uchar, 1> target_mask,
                               TransformCuda transform_source_to_target,
                               PinholeCameraIntrinsicCuda intrinsic);

}  // namespace cuda
}  // namespace open3d
