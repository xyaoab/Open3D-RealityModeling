//
// Created by akash on 2020-05-20
//

#include "SegmentationCuda.h"

namespace open3d {
namespace cuda {

ImageCuda<uchar, 1> SegmentationCuda::ComputeEdgeMap(
        ImageCuda<float, 3> &src_vertex_map,
        ImageCuda<float, 3> &src_normal_map) {
    assert(src_vertex_map.width_ == src_normal_map.width_);
    assert(src_vertex_map.height_ == src_normal_map.height_);
    ImageCuda<uchar, 1> edge_map;
    bool success =
            edge_map.Create(src_vertex_map.width_, src_vertex_map.height_);
    if (success) {
        SegmentationKernelCaller::ComputeEdgeMap(src_vertex_map, src_normal_map,
                                                 edge_map);
    }
    return edge_map;
}

ImageCuda<uchar, 1> SegmentationCuda::TransformAndProject(
        ImageCuda<uchar, 1> &src_mask,
        ImageCuda<float, 1> &src_depth,
        TransformCuda &transform_source_to_target,
        PinholeCameraIntrinsicCuda &intrinsic) {
    ImageCuda<uchar, 1> target_mask;
    bool success = target_mask.Create(src_mask.width_, src_mask.height_, 0);
    if (success) {
        SegmentationKernelCaller::TransformAndProject(
                src_mask, src_depth, target_mask, transform_source_to_target,
                intrinsic);
    }
    return target_mask;
}

}  // namespace cuda
}  // namespace open3d
