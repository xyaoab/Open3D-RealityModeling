// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <tbb/concurrent_unordered_set.h>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/CPU/TBBHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/t/geometry/kernel/VoxelBlockGridImpl.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

using index_t = int;

struct Coord3i {
    Coord3i(index_t x, index_t y, index_t z) : x_(x), y_(y), z_(z) {}
    bool operator==(const Coord3i &other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    index_t x_;
    index_t y_;
    index_t z_;
};


struct Coord3iHash {
    size_t operator()(const Coord3i &k) const {
        static const size_t p0 = 73856093;
        static const size_t p1 = 19349669;
        static const size_t p2 = 83492791;

        return (static_cast<size_t>(k.x_) * p0) ^
               (static_cast<size_t>(k.y_) * p1) ^
               (static_cast<size_t>(k.z_) * p2);
    }
};

struct Coord5f {
    Coord5f() {}
    Coord5f(float x, float y, float z, float angle, index_t idx): 
            x_(x), y_(y), z_(z), angle_(angle), idx_(idx){}
    bool operator==(const Coord5f &other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_ && angle_ == other.angle_
        && idx_ == other.idx_;
    }

    float x_;
    float y_;
    float z_;
    float angle_;
    index_t idx_;
};


// for pointcloud, select grids along the ray direction
void PointCloudRayMarchingCPU(std::shared_ptr<core::HashMap>
                &hashmap,  // dummy for now, one pass insertion is faster
        const core::Tensor &points,
        const core::Tensor &pose,
        core::Tensor &voxel_block_coords,
        core::Tensor &block_pcd_index,
        index_t voxel_grid_resolution,
        float voxel_size,
		index_t step_size, // param for num of voxels along the ray
        index_t tangential_step_size, // param for expansion on the tangential plane
        float sdf_trunc) {

        core::Device device = core::Device("CPU:0");

        index_t resolution = voxel_grid_resolution;
        float block_size = voxel_size * resolution;

        index_t n = points.GetLength();
        const float *pcd_ptr = static_cast<const float *>(points.GetDataPtr());
        // const float *pcd_normals_ptr = static_cast<const float *>(pcd_normals.GetDataPtr());

        tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
        tbb::concurrent_unordered_map<Coord3i, Coord5f, Coord3iHash> hashmap_block2points;

        const float *origin_ptr = static_cast<const float *>(pose.GetDataPtr());
        float x_o = origin_ptr[0];
        float y_o = origin_ptr[1];
        float z_o = origin_ptr[2];

        // populate neighbor points for association
        const float tangential_step = voxel_size;
        index_t num_blocks = tangential_step_size * tangential_step_size * 4;
        core::Tensor neighbor_pts = core::Tensor({num_blocks,3}, core::Float32, device);
        float *neighbor_pts_ptr = static_cast<float *>(neighbor_pts.GetDataPtr());
        index_t cnt = 0;
        for (auto ii=-tangential_step_size;ii<tangential_step_size;ii++) {
            for (auto jj=-tangential_step_size;jj<tangential_step_size;jj++) {
                neighbor_pts_ptr[cnt*3 + 0] = ii*tangential_step;
                neighbor_pts_ptr[cnt*3 + 1] = jj*tangential_step;
                neighbor_pts_ptr[cnt*3 + 2] = 0;
                cnt++;
            }
        }

        // for each xyz point
        core::ParallelFor(device, n, [&](index_t workload_idx) {

        // point in map frame
        float x = pcd_ptr[3 * workload_idx + 0];
        float y = pcd_ptr[3 * workload_idx + 1];
        float z = pcd_ptr[3 * workload_idx + 2];

        // Marching Ray Direction
        float x_d = x - x_o;
        float y_d = y - y_o;
        float z_d = z - z_o;
        float d = std::sqrt(x_d * x_d + y_d * y_d + z_d * z_d);
        std::vector<float> unit_normal = std::vector<float>{x_d/d, y_d/d, z_d/d};
        float denominator = std::sqrt(unit_normal[0] * unit_normal[0]
                                    + unit_normal[1] * unit_normal[1]);

        std::vector<float> fraction_ptr = std::vector<float>{unit_normal[0]/denominator, 
                                        unit_normal[1]/denominator, unit_normal[2]/denominator};
        // get plane rotation matrix
        core::Tensor rotation = core::Tensor::Init<float>(
                {{fraction_ptr[1], -fraction_ptr[0], 0.}, 
                {fraction_ptr[0] * unit_normal[2], 
                    fraction_ptr[1] * unit_normal[2], -denominator},
                {fraction_ptr[0], unit_normal[1], unit_normal[2]}}, device);

		const float t_min = (d - sdf_trunc) / d; //max(d - sdf_trunc, 0.0f) / d;
		const float t_max = (d + sdf_trunc) / d; // min(d + sdf_trunc, depth_max) / d;
		const float t_step = (t_max - t_min) / step_size;

        core::Tensor rotated_neighbor_pts = rotation.Matmul(neighbor_pts.T()).T();
        float *rotated_neighbors_ptr = static_cast<float *>(rotated_neighbor_pts.GetDataPtr());

        float t = t_min;
        Coord5f current_pcd_coords{x, y, z, 0, -1};

        for (index_t step = 0; step <= step_size; ++step) {
            float x_f = x_o + t * x_d;
            float y_f = y_o + t * y_d;
            float z_f = z_o + t * z_d;

            for (index_t idx = 0; idx<rotated_neighbor_pts.GetShape(0); idx++){
                index_t x_neighbor = static_cast<index_t>(
				    std::floor((x_f + rotated_neighbors_ptr[idx*3 + 0]) / block_size));
                index_t y_neighbor = static_cast<index_t>(
				    std::floor((y_f + rotated_neighbors_ptr[idx*3 + 1]) / block_size));
                index_t z_neighbor = static_cast<index_t>(
				    std::floor((z_f + rotated_neighbors_ptr[idx*3 + 2]) / block_size));
                
                const Coord3i neighbor_block_coords{x_neighbor, y_neighbor, z_neighbor};
                set.emplace(x_neighbor, y_neighbor, z_neighbor);

                std::vector<float> block_dir = std::vector<float>{static_cast<float>(x_neighbor) - x_o, 
                                                static_cast<float>(y_neighbor) - y_o,
                                                static_cast<float>(z_neighbor) - z_o};
                float block_dir_norm = std::sqrt(block_dir[0] * block_dir[0]
                                        + block_dir[1] * block_dir[1]
                                        + block_dir[2] * block_dir[2]);
                float current_angle = std::fabs((block_dir[0] * unit_normal[0]
                                        + block_dir[1] * unit_normal[1]
                                        + block_dir[2] * unit_normal[2]) / block_dir_norm);
                current_pcd_coords.angle_ = current_angle;
                current_pcd_coords.idx_ = workload_idx;
                // update weighted average pcd 3d points
                if ((hashmap_block2points.count(neighbor_block_coords)!= 0 
                    && hashmap_block2points[neighbor_block_coords].angle_ < current_angle)
                    || hashmap_block2points.count(neighbor_block_coords)==0) {

                    hashmap_block2points[neighbor_block_coords] = current_pcd_coords;
                }
            }

			t += t_step;
        }
        });

        index_t block_count = set.size();
        if (block_count == 0) {
			utility::LogError(
				"No block is touched in TSDF volume, abort integration. Please "
				"check specified parameters, "
				"especially depth_scale and voxel_size");
        }


        voxel_block_coords =
			core::Tensor({block_count, 3}, core::Int32, points.GetDevice());
        index_t *block_coords_ptr =
			static_cast<index_t *>(voxel_block_coords.GetDataPtr());

        block_pcd_index =
			core::Tensor({block_count, 1}, core::Int32, points.GetDevice());
        index_t *block_pcd_index_ptr =
			static_cast<index_t *>(block_pcd_index.GetDataPtr());

        index_t count = 0;
        for (auto it = set.begin(); it != set.end(); ++it, ++count) {
			index_t offset = count * 3;
            index_t blockX = static_cast<index_t>(it->x_);
            index_t blockY = static_cast<index_t>(it->y_);
            index_t blockZ = static_cast<index_t>(it->z_);
            Coord3i block = Coord3i{blockX, blockY, blockZ};
            // float pcdX = static_cast<float>(hashmap_block2points[block].x_);
            // float pcdY = static_cast<float>(hashmap_block2points[block].y_);
            // float pcdZ = static_cast<float>(hashmap_block2points[block].z_);
            
            index_t pcdIdx = static_cast<index_t>(hashmap_block2points[block].idx_);


			block_coords_ptr[offset + 0] = blockX;
			block_coords_ptr[offset + 1] = blockY;
			block_coords_ptr[offset + 2] = blockZ;

            block_pcd_index_ptr[count] = pcdIdx;
            // block_pcd_coords_ptr[offset + 0] = pcdX;
            // block_pcd_coords_ptr[offset + 1] = pcdY;
            // block_pcd_coords_ptr[offset + 2] = pcdZ;
            // block_pcd_normals_ptr[offset + 0] = pcd_normals_ptr[3 * pcdIdx + 0];
            // block_pcd_normals_ptr[offset + 1] = pcd_normals_ptr[3 * pcdIdx + 1];
            // block_pcd_normals_ptr[offset + 2] = pcd_normals_ptr[3 * pcdIdx + 2];
        }
}


void PointCloudTouchCPU(
        std::shared_ptr<core::HashMap>
                &hashmap,  // dummy for now, one pass insertion is faster
        const core::Tensor &points,
        core::Tensor &voxel_block_coords,
        index_t voxel_grid_resolution,
        float voxel_size,
        float sdf_trunc) {
    index_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    index_t n = points.GetLength();
    const float *pcd_ptr = static_cast<const float *>(points.GetDataPtr());

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    core::ParallelFor(core::Device("CPU:0"), n, [&](index_t workload_idx) {
        float x = pcd_ptr[3 * workload_idx + 0];
        float y = pcd_ptr[3 * workload_idx + 1];
        float z = pcd_ptr[3 * workload_idx + 2];

        index_t xb_lo =
                static_cast<index_t>(std::floor((x - sdf_trunc) / block_size));
        index_t xb_hi =
                static_cast<index_t>(std::floor((x + sdf_trunc) / block_size));
        index_t yb_lo =
                static_cast<index_t>(std::floor((y - sdf_trunc) / block_size));
        index_t yb_hi =
                static_cast<index_t>(std::floor((y + sdf_trunc) / block_size));
        index_t zb_lo =
                static_cast<index_t>(std::floor((z - sdf_trunc) / block_size));
        index_t zb_hi =
                static_cast<index_t>(std::floor((z + sdf_trunc) / block_size));
        for (index_t xb = xb_lo; xb <= xb_hi; ++xb) {
            for (index_t yb = yb_lo; yb <= yb_hi; ++yb) {
                for (index_t zb = zb_lo; zb <= zb_hi; ++zb) {
                    set.emplace(xb, yb, zb);
                }
            }
        }
    });

    index_t block_count = set.size();
    if (block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, abort integration. Please "
                "check specified parameters, "
                "especially depth_scale and voxel_size");
    }

    voxel_block_coords =
            core::Tensor({block_count, 3}, core::Int32, points.GetDevice());
    index_t *block_coords_ptr =
            static_cast<index_t *>(voxel_block_coords.GetDataPtr());
    index_t count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        index_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<index_t>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<index_t>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<index_t>(it->z_);
    }
}

void DepthTouchCPU(std::shared_ptr<core::HashMap> &hashmap,
                   const core::Tensor &depth,
                   const core::Tensor &intrinsic,
                   const core::Tensor &extrinsic,
                   core::Tensor &voxel_block_coords,
                   index_t voxel_grid_resolution,
                   float voxel_size,
                   float sdf_trunc,
                   float depth_scale,
                   float depth_max,
                   index_t stride) {
    core::Device device = depth.GetDevice();
    NDArrayIndexer depth_indexer(depth, 2);
    core::Tensor pose = t::geometry::InverseTransformation(extrinsic);
    TransformIndexer ti(intrinsic, pose, 1.0f);

    // Output
    index_t rows_strided = depth_indexer.GetShape(0) / stride;
    index_t cols_strided = depth_indexer.GetShape(1) / stride;
    index_t n = rows_strided * cols_strided;

    index_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::ParallelFor(device, n, [&](index_t workload_idx) {
            index_t y = (workload_idx / cols_strided) * stride;
            index_t x = (workload_idx % cols_strided) * stride;

            float d = *depth_indexer.GetDataPtr<scalar_t>(x, y) / depth_scale;
            if (d > 0 && d < depth_max) {
                float x_c = 0, y_c = 0, z_c = 0;
                ti.Unproject(static_cast<float>(x), static_cast<float>(y), 1.0,
                             &x_c, &y_c, &z_c);
                float x_g = 0, y_g = 0, z_g = 0;
                ti.RigidTransform(x_c, y_c, z_c, &x_g, &y_g, &z_g);

                // Origin
                float x_o = 0, y_o = 0, z_o = 0;
                ti.GetCameraPosition(&x_o, &y_o, &z_o);

                // Direction
                float x_d = x_g - x_o;
                float y_d = y_g - y_o;
                float z_d = z_g - z_o;

                const index_t step_size = 3;
                const float t_min = std::max(d - sdf_trunc, 0.0f);
                const float t_max = std::min(d + sdf_trunc, depth_max);
                const float t_step = (t_max - t_min) / step_size;

                float t = t_min;
                for (index_t step = 0; step <= step_size; ++step) {
                    index_t xb = static_cast<index_t>(
                            std::floor((x_o + t * x_d) / block_size));
                    index_t yb = static_cast<index_t>(
                            std::floor((y_o + t * y_d) / block_size));
                    index_t zb = static_cast<index_t>(
                            std::floor((z_o + t * z_d) / block_size));
                    set.emplace(xb, yb, zb);
                    t += t_step;
                }
            }
        });
    });

    index_t block_count = set.size();
    if (block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, abort integration. Please "
                "check specified parameters, "
                "especially depth_scale and voxel_size");
    }

    voxel_block_coords = core::Tensor({block_count, 3}, core::Int32, device);
    index_t *block_coords_ptr = voxel_block_coords.GetDataPtr<index_t>();
    index_t count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        index_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<index_t>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<index_t>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<index_t>(it->z_);
    }
}

#define FN_ARGUMENTS                                                          \
    const core::Tensor &depth, const core::Tensor &color,                     \
            const core::Tensor &indices, const core::Tensor &block_keys,      \
            TensorMap &value_tensor_map, const core::Tensor &depth_intrinsic, \
            const core::Tensor &color_intrinsic,                              \
            const core::Tensor &extrinsic, index_t resolution,                \
            float voxel_size, float sdf_trunc, float depth_scale,             \
            float depth_max

template void IntegrateCPU<uint16_t, uint8_t, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCPU<uint16_t, uint8_t, float, float, float>(
        FN_ARGUMENTS);
template void IntegrateCPU<float, float, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCPU<float, float, float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                           \
    std::shared_ptr<core::HashMap> &hashmap, const TensorMap &block_value_map, \
            const core::Tensor &range_map, TensorMap &renderings_map,          \
            const core::Tensor &intrinsic, const core::Tensor &extrinsic,      \
            index_t h, index_t w, index_t block_resolution, float voxel_size,  \
            float depth_scale, float depth_min, float depth_max,               \
            float weight_threshold, float trunc_voxel_multiplier,              \
            int range_map_down_factor

template void RayCastCPU<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void RayCastCPU<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                           \
    const core::Tensor &block_indices, const core::Tensor &nb_block_indices,   \
            const core::Tensor &nb_block_masks,                                \
            const core::Tensor &block_keys, const TensorMap &block_value_map,  \
            core::Tensor &points, core::Tensor &normals, core::Tensor &colors, \
            index_t block_resolution, float voxel_size,                        \
            float weight_threshold, index_t &valid_size

template void ExtractPointCloudCPU<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void ExtractPointCloudCPU<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                          \
    const core::Tensor &block_indices, const core::Tensor &inv_block_indices, \
            const core::Tensor &nb_block_indices,                             \
            const core::Tensor &nb_block_masks,                               \
            const core::Tensor &block_keys, const TensorMap &block_value_map, \
            core::Tensor &vertices, core::Tensor &triangles,                  \
            core::Tensor &vertex_normals, core::Tensor &vertex_colors,        \
            index_t block_resolution, float voxel_size,                       \
            float weight_threshold, index_t &vertex_count

template void ExtractTriangleMeshCPU<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void ExtractTriangleMeshCPU<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
