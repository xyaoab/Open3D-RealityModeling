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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/CUDA/StdGPUHashBackend.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/HashMap.h"
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

struct Coord3i {
    OPEN3D_HOST_DEVICE Coord3i(index_t x, index_t y, index_t z)
        : x_(x), y_(y), z_(z) {}
    OPEN3D_HOST_DEVICE bool operator==(const Coord3i &other) const {
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


template <typename src_t, typename dst_t>
static void OPEN3D_DEVICE CUDAEqElementKernel(const void* lhs,
                                                   const void* rhs,
                                                   void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) == *static_cast<const src_t*>(rhs));
}


void PointCloudRayMarchingCUDA(std::shared_ptr<core::HashMap>
                &hashmap,
        const core::Tensor &points,
        const core::Tensor &pcd_normals,
        const core::Tensor &extrinsic,
        core::Tensor &voxel_block_coords,
		core::Tensor &block_pcd_coords,
        core::Tensor &block_pcd_normals,
        index_t voxel_grid_resolution,
        float voxel_size,
		index_t step_size,
		index_t tangential_step_size,
        float sdf_trunc){

        core::Device device = points.GetDevice();
        // sensor origin
        core::Tensor pose = t::geometry::InverseTransformation(extrinsic);
        index_t resolution = voxel_grid_resolution;
        float block_size = voxel_size * resolution;

        index_t n = points.GetLength();
        const float *pcd_ptr = static_cast<const float *>(points.GetDataPtr());
        const float *pcd_normals_ptr = static_cast<const float *>(pcd_normals.GetDataPtr());
        const float *origin_ptr = static_cast<const float *>(pose.GetDataPtr());
        float x_o = origin_ptr[0*4+3];
        float y_o = origin_ptr[1*4+3];
        float z_o = origin_ptr[2*4+3];

		const index_t num_blocks = tangential_step_size * tangential_step_size * 4;
        const index_t est_multipler_factor = (step_size + 1) *  num_blocks;

        core::Tensor block_coordi({est_multipler_factor * n, 3}, core::Int32, device);
    	index_t *block_coordi_ptr = static_cast<index_t *>(block_coordi.GetDataPtr());
    
		// save the block - pcd_index association [before hashing]
		core::Tensor block_pcd_lookup({est_multipler_factor * n, 1}, core::Int32, device);
    	index_t *block_pcd_lookup_ptr =
            static_cast<index_t *>(block_pcd_lookup.GetDataPtr());

        core::Tensor block_angle_lookup({est_multipler_factor * n, 1}, core::Float32, device);
    	float *block_angle_lookup_ptr =
            static_cast<float *>(block_angle_lookup.GetDataPtr());

        core::Tensor count(std::vector<index_t>{0}, {}, core::Int32, device);
    	index_t *count_ptr = static_cast<index_t *>(count.GetDataPtr());

         // populate neighbor points for association
        const float tangential_step = voxel_size;

        core::Tensor neighbor_pts_host({num_blocks, 3}, core::Dtype::Float32, core::Device("CPU:0"));
        float *neighbor_pts_host_ptr = static_cast<float *>(neighbor_pts_host.GetDataPtr());

		index_t cnt = 0;
		for (auto ii=-tangential_step_size;ii<tangential_step_size;ii++) {
            for (auto jj=-tangential_step_size;jj<tangential_step_size;jj++) {	
				neighbor_pts_host_ptr[cnt * 3 + 0] = ii*tangential_step;
                neighbor_pts_host_ptr[cnt * 3 + 1] = jj*tangential_step;
                neighbor_pts_host_ptr[cnt * 3 + 2] = 0;
				cnt++;
            }
        }

		// move from cpu host to gpu device
		core::Tensor neighbor_pts = neighbor_pts_host.To(device);
        float *neighbor_pts_ptr = static_cast<float *>(neighbor_pts.GetDataPtr());

        // for each xyz point
        core::ParallelFor(hashmap->GetDevice(), n,
                      [=] OPEN3D_DEVICE(index_t workload_idx) {
			// point in map frame
			float x = pcd_ptr[3 * workload_idx + 0];
			float y = pcd_ptr[3 * workload_idx + 1];
			float z = pcd_ptr[3 * workload_idx + 2];
			

			// Marching Ray Direction
			float x_d = x - x_o, y_d = y - y_o, z_d = z - z_o;
			float d = sqrtf(x_d * x_d + y_d * y_d + z_d * z_d);

			// unit_normal
			float unit_normal_x = x_d / d,
				unit_normal_y = y_d / d,
				unit_normal_z = z_d / d;
			float denominator = std::sqrt(unit_normal_x * unit_normal_x
										+ unit_normal_y * unit_normal_y);

			float fraction_x = unit_normal_x / denominator,
				fraction_y = unit_normal_y / denominator;

			const float t_min = (d - sdf_trunc) / d; //max(d - sdf_trunc, 0.0f) / d;
			const float t_max = (d + sdf_trunc) /  d ; // min(d + sdf_trunc, depth_max) / d;
			const float t_step = (t_max - t_min) / step_size;

			float t = t_min;

			for (index_t step = 0; step <= step_size; ++step) {

				float x_f = x_o + t * x_d;
				float y_f = y_o + t * y_d;
				float z_f = z_o + t * z_d;

				for (index_t ii = 0; ii<num_blocks; ii++){
					float x_g = 0, y_g = 0, z_g = 0;
					float x_in = neighbor_pts_ptr[ii*3 + 0], 
						y_in = neighbor_pts_ptr[ii*3 + 1],
						z_in = neighbor_pts_ptr[ii*3 + 2];
					
					x_g = x_in * fraction_y + y_in * (-fraction_x) + z_in * 0;
					y_g = x_in * (fraction_x * unit_normal_z)  
						+ y_in * (fraction_y * unit_normal_z) +
						z_in * (-denominator);
					z_g = x_in * fraction_x + y_in * unit_normal_y +
						z_in * unit_normal_z;

					index_t x_neighbor = static_cast<index_t>(
						std::floor((x_f + x_g) / block_size));
					index_t y_neighbor = static_cast<index_t>(
						std::floor((y_f + y_g) / block_size));
					index_t z_neighbor = static_cast<index_t>(
						std::floor((z_f + z_g) / block_size));
					index_t idx = atomicAdd(count_ptr, 1);
					
					// saving block coordi before hasing
					block_coordi_ptr[3 * idx + 0] = x_neighbor;
					block_coordi_ptr[3 * idx + 1] = y_neighbor;
					block_coordi_ptr[3 * idx + 2] = z_neighbor;

                    // change herustics to using distance
                    float block_pcd_dist_x = static_cast<float>(x_neighbor) - x, 
						block_pcd_dist_y = static_cast<float>(y_neighbor) - y,
						block_pcd_dist_z = static_cast<float>(z_neighbor) - z;
                    float current_distance =  std::sqrt(block_pcd_dist_x * block_pcd_dist_x
                                            + block_pcd_dist_y * block_pcd_dist_y
                                            + block_pcd_dist_z * block_pcd_dist_z);
					block_pcd_lookup_ptr[idx] = workload_idx;
					block_angle_lookup_ptr[idx] = current_distance;
		
				}
				t += t_step;
			}
		});

		index_t total_block_count = count.Item<index_t>();

		if (total_block_count == 0) {
			utility::LogError(
					"[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
					"abort integration. Please check specified parameters, "
					"especially depth_scale and voxel_size");
		}
        // Step 1: Activation / Insertion into hashmap
		block_coordi = block_coordi.Slice(0, 0, total_block_count);
		core::Tensor block_buf_indices, block_masks;
		hashmap->Activate(block_coordi, block_buf_indices, block_masks);
        index_t num_unique_blocks = hashmap->Size();

		voxel_block_coords = block_coordi.IndexGet({block_masks});
        block_pcd_coords = core::Tensor(voxel_block_coords.GetShape(), core::Float32, device);
        float *block_pcd_coords_ptr = static_cast<float *>(block_pcd_coords.GetDataPtr());
        
        block_pcd_normals = core::Tensor(voxel_block_coords.GetShape(), core::Float32, device); 
        float *block_pcd_normals_ptr = static_cast<float *>(block_pcd_normals.GetDataPtr());
        // Step 2: All keys shall reside in the hashmap
        // to get depulicated key location in the hashmap
        hashmap->Find(block_coordi, block_buf_indices, block_masks);
        // assert
        if (!block_masks.All()){
            utility::LogError(
					"[PointCloudRayMarchingCUDA] Hashmap find has missing keys.");
        }
        if (block_buf_indices.Max({0}, true).Item<index_t>() != num_unique_blocks-1 ){
            utility::LogError(
					"[PointCloudRayMarchingCUDA] block_buf_indices size incorrect.");
        }
        if (num_unique_blocks != voxel_block_coords.GetShape(0)){
            utility::LogError(
					"[PointCloudRayMarchingCUDA] hashmap size incorrect.");
        }

        // Init tensor to be updated and written into hashmap value
        std::vector<float> vec_val(num_unique_blocks, 1000.f);
        std::vector<index_t> vec_index(num_unique_blocks, 0);

        core::Tensor result_value(vec_val, {num_unique_blocks, 1}, core::Float32, device);
        float *result_value_ptr = static_cast<float *>(result_value.GetDataPtr());

        core::Tensor result_index(vec_index, {num_unique_blocks, 1}, core::Int32, device);
        index_t *result_index_ptr = static_cast<index_t *>(result_index.GetDataPtr());

        index_t *block_buf_indices_ptr = static_cast<index_t *>(block_buf_indices.GetDataPtr());

        // Pass 1: do elementwise max operation for angle herustics
         core::ParallelFor(hashmap->GetDevice(), total_block_count,
                      [=] OPEN3D_DEVICE(index_t workload_idx){
            index_t hash_idx = block_buf_indices_ptr[workload_idx];
            atomicMinf(&result_value_ptr[hash_idx], block_angle_lookup_ptr[workload_idx]);
            });

        // Pass 2: find the argmax angle to pcd
        core::ParallelFor(hashmap->GetDevice(), total_block_count,
                      [=] OPEN3D_DEVICE(index_t workload_idx){
            index_t hash_idx = block_buf_indices_ptr[workload_idx];
            // finding the index with the largest angle with some precision tolerance
            if (result_value_ptr[hash_idx] > block_angle_lookup_ptr[workload_idx] - 1e-6){
                result_index_ptr[hash_idx] = block_pcd_lookup_ptr[workload_idx];
            }
            });

        // Pass 3: prepare results tensors for association of pcd index
        core::Tensor voxel_block_buf_indices, voxel_block_masks;
        hashmap->Find(voxel_block_coords, voxel_block_buf_indices, voxel_block_masks);

        index_t *voxel_block_buf_indices_ptr = static_cast<index_t *>(voxel_block_buf_indices.GetDataPtr());
        core::ParallelFor(hashmap->GetDevice(), voxel_block_coords.GetShape(0),
                      [=] OPEN3D_DEVICE(index_t workload_idx){

            index_t hash_idx = voxel_block_buf_indices_ptr[workload_idx];
            index_t pcd_idx = result_index_ptr[hash_idx];
            float pcdX = pcd_ptr[3 * pcd_idx + 0];
            float pcdY = pcd_ptr[3 * pcd_idx + 1];
            float pcdZ = pcd_ptr[3 * pcd_idx + 2];

            block_pcd_coords_ptr[3 * workload_idx + 0] = pcdX;
            block_pcd_coords_ptr[3 * workload_idx + 1] = pcdY;
            block_pcd_coords_ptr[3 * workload_idx + 2] = pcdZ;

            block_pcd_normals_ptr[3 * workload_idx + 0] = pcd_normals_ptr[3 * pcd_idx + 0];
            block_pcd_normals_ptr[3 * workload_idx + 1] = pcd_normals_ptr[3 * pcd_idx + 1];
            block_pcd_normals_ptr[3 * workload_idx + 2] = pcd_normals_ptr[3 * pcd_idx + 2];

            });
}

void PointCloudTouchCUDA(std::shared_ptr<core::HashMap> &hashmap,
                         const core::Tensor &points,
                         core::Tensor &voxel_block_coords,
                         index_t voxel_grid_resolution,
                         float voxel_size,
                         float sdf_trunc) {
    index_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    index_t n = points.GetLength();
    const float *pcd_ptr = static_cast<const float *>(points.GetDataPtr());

    core::Device device = points.GetDevice();
    core::Tensor block_coordi({8 * n, 3}, core::Int32, device);
    index_t *block_coordi_ptr =
            static_cast<index_t *>(block_coordi.GetDataPtr());
    core::Tensor count(std::vector<index_t>{0}, {}, core::Int32, device);
    index_t *count_ptr = static_cast<index_t *>(count.GetDataPtr());

    core::ParallelFor(hashmap->GetDevice(), n,
                      [=] OPEN3D_DEVICE(index_t workload_idx) {
                          float x = pcd_ptr[3 * workload_idx + 0];
                          float y = pcd_ptr[3 * workload_idx + 1];
                          float z = pcd_ptr[3 * workload_idx + 2];

                          index_t xb_lo = static_cast<index_t>(
                                  floorf((x - sdf_trunc) / block_size));
                          index_t xb_hi = static_cast<index_t>(
                                  floorf((x + sdf_trunc) / block_size));
                          index_t yb_lo = static_cast<index_t>(
                                  floorf((y - sdf_trunc) / block_size));
                          index_t yb_hi = static_cast<index_t>(
                                  floorf((y + sdf_trunc) / block_size));
                          index_t zb_lo = static_cast<index_t>(
                                  floorf((z - sdf_trunc) / block_size));
                          index_t zb_hi = static_cast<index_t>(
                                  floorf((z + sdf_trunc) / block_size));

                          for (index_t xb = xb_lo; xb <= xb_hi; ++xb) {
                              for (index_t yb = yb_lo; yb <= yb_hi; ++yb) {
                                  for (index_t zb = zb_lo; zb <= zb_hi; ++zb) {
                                      index_t idx = atomicAdd(count_ptr, 1);
                                      block_coordi_ptr[3 * idx + 0] = xb;
                                      block_coordi_ptr[3 * idx + 1] = yb;
                                      block_coordi_ptr[3 * idx + 2] = zb;
                                  }
                              }
                          }
                      });

    index_t total_block_count = count.Item<index_t>();

    if (total_block_count == 0) {
        utility::LogError(
                "[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Tensor block_buf_indices, block_masks;
    hashmap->Activate(block_coordi.Slice(0, 0, count.Item<index_t>()),
                      block_buf_indices, block_masks);
    voxel_block_coords = block_coordi.IndexGet({block_masks});
}

void DepthTouchCUDA(std::shared_ptr<core::HashMap> &hashmap,
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

    const index_t step_size = 3;
    const index_t est_multipler_factor = (step_size + 1);

    static core::Tensor block_coordi;
    if (block_coordi.GetLength() != est_multipler_factor * n) {
        block_coordi = core::Tensor({est_multipler_factor * n, 3},
                                    core::Dtype::Int32, device);
    }

    // Counter
    core::Tensor count(std::vector<index_t>{0}, {1}, core::Dtype::Int32,
                       device);
    index_t *count_ptr = count.GetDataPtr<index_t>();
    index_t *block_coordi_ptr = block_coordi.GetDataPtr<index_t>();

    index_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
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

                const float t_min = max(d - sdf_trunc, 0.0);
                const float t_max = min(d + sdf_trunc, depth_max);
                const float t_step = (t_max - t_min) / step_size;

                float t = t_min;
                index_t idx = OPEN3D_ATOMIC_ADD(count_ptr, (step_size + 1));
                for (index_t step = 0; step <= step_size; ++step) {
                    index_t offset = (step + idx) * 3;

                    index_t xb = static_cast<index_t>(
                            floorf((x_o + t * x_d) / block_size));
                    index_t yb = static_cast<index_t>(
                            floorf((y_o + t * y_d) / block_size));
                    index_t zb = static_cast<index_t>(
                            floorf((z_o + t * z_d) / block_size));

                    block_coordi_ptr[offset + 0] = xb;
                    block_coordi_ptr[offset + 1] = yb;
                    block_coordi_ptr[offset + 2] = zb;

                    t += t_step;
                }
            }
        });
    });

    index_t total_block_count = static_cast<index_t>(count[0].Item<index_t>());
    if (total_block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }

    total_block_count = std::min(total_block_count,
                                 static_cast<index_t>(hashmap->GetCapacity()));
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Tensor block_addrs, block_masks;
    hashmap->Activate(block_coordi, block_addrs, block_masks);

    // Customized IndexGet (generic version too slow)
    voxel_block_coords =
            core::Tensor({hashmap->Size(), 3}, core::Int32, device);
    index_t *voxel_block_coord_ptr = voxel_block_coords.GetDataPtr<index_t>();
    bool *block_masks_ptr = block_masks.GetDataPtr<bool>();
    count[0] = 0;
    core::ParallelFor(device, total_block_count,
                      [=] OPEN3D_DEVICE(index_t workload_idx) {
                          if (block_masks_ptr[workload_idx]) {
                              index_t idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
                              index_t offset_lhs = 3 * idx;
                              index_t offset_rhs = 3 * workload_idx;
                              voxel_block_coord_ptr[offset_lhs + 0] =
                                      block_coordi_ptr[offset_rhs + 0];
                              voxel_block_coord_ptr[offset_lhs + 1] =
                                      block_coordi_ptr[offset_rhs + 1];
                              voxel_block_coord_ptr[offset_lhs + 2] =
                                      block_coordi_ptr[offset_rhs + 2];
                          }
                      });
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

#define FN_ARGUMENTS                                                      \
    const core::Tensor &depth, const core::Tensor &color,                 \
            const core::Tensor &indices, const core::Tensor &block_keys,  \
            TensorMap &block_values, const core::Tensor &depth_intrinsic, \
            const core::Tensor &color_intrinsic,                          \
            const core::Tensor &extrinsic, index_t resolution,            \
            float voxel_size, float sdf_trunc, float depth_scale,         \
            float depth_max

template void IntegrateCUDA<uint16_t, uint8_t, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCUDA<uint16_t, uint8_t, float, float, float>(
        FN_ARGUMENTS);
template void IntegrateCUDA<float, float, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCUDA<float, float, float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                           \
    std::shared_ptr<core::HashMap> &hashmap, const TensorMap &block_value_map, \
            const core::Tensor &range_map, TensorMap &renderings_map,          \
            const core::Tensor &intrinsic, const core::Tensor &extrinsic,      \
            index_t h, index_t w, index_t block_resolution, float voxel_size,  \
            float depth_scale, float depth_min, float depth_max,               \
            float weight_threshold, float trunc_voxel_multiplier,              \
            int range_map_down_factor

template void RayCastCUDA<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void RayCastCUDA<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                           \
    const core::Tensor &block_indices, const core::Tensor &nb_block_indices,   \
            const core::Tensor &nb_block_masks,                                \
            const core::Tensor &block_keys, const TensorMap &block_value_map,  \
            core::Tensor &points, core::Tensor &normals, core::Tensor &colors, \
            index_t block_resolution, float voxel_size,                        \
            float weight_threshold, index_t &valid_size

template void ExtractPointCloudCUDA<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void ExtractPointCloudCUDA<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

void ExtractTriangleMeshCUDA(const core::Tensor &block_indices,
                             const core::Tensor &inv_block_indices,
                             const core::Tensor &nb_block_indices,
                             const core::Tensor &nb_block_masks,
                             const core::Tensor &block_keys,
                             const std::vector<core::Tensor> &block_values,
                             core::Tensor &vertices,
                             core::Tensor &triangles,
                             core::Tensor &vertex_normals,
                             core::Tensor &vertex_colors,
                             index_t block_resolution,
                             float voxel_size,
                             float weight_threshold,
                             index_t &vertex_count);

#define FN_ARGUMENTS                                                          \
    const core::Tensor &block_indices, const core::Tensor &inv_block_indices, \
            const core::Tensor &nb_block_indices,                             \
            const core::Tensor &nb_block_masks,                               \
            const core::Tensor &block_keys, const TensorMap &block_value_map, \
            core::Tensor &vertices, core::Tensor &triangles,                  \
            core::Tensor &vertex_normals, core::Tensor &vertex_colors,        \
            index_t block_resolution, float voxel_size,                       \
            float weight_threshold, index_t &vertex_count

template void ExtractTriangleMeshCUDA<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void ExtractTriangleMeshCUDA<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
