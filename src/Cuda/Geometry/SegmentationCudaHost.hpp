//
// Created by akash on 2020-05-20
//

#include "SegmentationCuda.h"

namespace open3d {
namespace cuda {

ImageCuda<float, 1> SegmentationCuda::ComputeEdgeMap(ImageCuda<float, 3> &src_vertex_map,
                                   ImageCuda<float, 3> &src_normal_map) {
    assert(src_vertex_map.width_ == src_normal_map.width_);
    assert(src_vertex_map.height_ == src_normal_map.height_);
    ImageCuda<float, 1> edge_map;
    bool success =
            edge_map.Create(src_vertex_map.width_, src_vertex_map.height_);
    if (success) {
        SegmentationKernelCaller::ComputeEdgeMap(src_vertex_map, src_normal_map,
                                                 edge_map);
    }
    return edge_map;
}

} //namespace cuda
} //namespace open3d
