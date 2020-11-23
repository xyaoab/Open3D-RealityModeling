// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/t/geometry/VoxelGrid.h"

#include <Eigen/Dense>

#include "open3d/core/SparseTensorList.h"
#include "open3d/core/TensorList.h"
#include "open3d/core/kernel/SpecialOp.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace geometry {
using namespace core;

VoxelGrid::VoxelGrid(float voxel_size,
                     float sdf_trunc,
                     int64_t resolution,
                     int64_t capacity,
                     const Device &device)
    : Geometry(Geometry::GeometryType::VoxelGrid, 3),
      voxel_size_(voxel_size),
      sdf_trunc_(sdf_trunc),
      resolution_(resolution),
      capacity_(capacity),
      device_(device) {
    Dtype key_dtype(Dtype::DtypeCode::Object, 3 * sizeof(int64_t), "key3d");
    Dtype val_dtype(Dtype::DtypeCode::Object,
                    2 * resolution * resolution * resolution * sizeof(float),
                    "valtensor3d");
    hashmap_ =
            std::make_shared<Hashmap>(capacity, key_dtype, val_dtype, device);
}

void VoxelGrid::Integrate(const Image &depth,
                          const Tensor &intrinsic,
                          const Tensor &extrinsic) {
    /// Unproject

    utility::Timer timer;
    timer.Start();
    Tensor vertex_map = depth.Unproject(intrinsic);
    Tensor pcd_map = vertex_map.View({3, 480 * 640});
    timer.Stop();
    utility::LogInfo("Unproject takes {}", timer.GetDuration());

    /// Inverse is currently not available...
    Tensor pose = extrinsic.Inverse();
    PointCloud pcd(TensorList::FromTensor(pcd_map.T()));
    pcd.Transform(pose);

    timer.Start();
    PointCloud pcd_down = pcd.VoxelDownSample(voxel_size_ * resolution_);
    timer.Stop();
    utility::LogInfo("Downsample takes {}", timer.GetDuration());

    Tensor coords = pcd_down.GetPoints().AsTensor().To(Dtype::Int64);
    SizeVector coords_shape = coords.GetShape();
    int64_t N = coords_shape[0];

    void *iterators = MemoryManager::Malloc(sizeof(iterator_t) * N, device_);
    void *masks = MemoryManager::Malloc(sizeof(bool) * N, device_);
    void *blob_indices = MemoryManager::Malloc(sizeof(addr_t) * N, device_);

    timer.Start();
    hashmap_->Activate(static_cast<void *>(coords.GetBlob()->GetDataPtr()),
                       static_cast<iterator_t *>(iterators),
                       static_cast<bool *>(masks),
                       static_cast<addr_t *>(blob_indices), N);
    timer.Stop();
    utility::LogInfo("Activate takes {}", timer.GetDuration());

    timer.Start();
    hashmap_->Find(static_cast<void *>(coords.GetBlob()->GetDataPtr()),
                   static_cast<iterator_t *>(iterators),
                   static_cast<bool *>(masks),
                   static_cast<addr_t *>(blob_indices), N);
    timer.Stop();
    utility::LogInfo("Find takes {}", timer.GetDuration());

    utility::LogInfo("Active entries = {}", N);

    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_tl(static_cast<void **>(iterators), N, true,
                               {shape, shape}, {Dtype::Float32, Dtype::Float32},
                               device_);
    Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);
    Tensor sdf_trunc(std::vector<float>{sdf_trunc_}, {1}, Dtype::Float32);

    Tensor output_dummy;
    timer.Start();
    kernel::SpecialOpEW({depth.AsTensor().Permute({2, 0, 1}).Contiguous(),
                         intrinsic, extrinsic, voxel_size, sdf_trunc},
                        {}, output_dummy, sparse_tl,
                        kernel::SpecialOpCode::Integrate);
    timer.Stop();
    utility::LogInfo("Kernel takes {}", timer.GetDuration());
    utility::LogInfo("[VoxelGrid] Kernel launch finished");

    // Then manipulate iterators to integrate!
    MemoryManager::Free(iterators, coords.GetDevice());
    MemoryManager::Free(masks, coords.GetDevice());
    MemoryManager::Free(blob_indices, coords.GetDevice());

    utility::LogInfo("Hashmap size = {}", hashmap_->Size());
}

std::pair<SparseTensorList, Tensor> VoxelGrid::ExtractNearestNeighbors() {
    int64_t n = hashmap_->Size();
    void *block_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *blob_indices = MemoryManager::Malloc(sizeof(addr_t) * n, device_);

    hashmap_->GetIterators(static_cast<iterator_t *>(block_iterators),
                           static_cast<addr_t *>(blob_indices));

    Tensor keys({n, 3}, Dtype::Int64, device_);
    hashmap_->UnpackIterators(static_cast<iterator_t *>(block_iterators),
                              nullptr, keys.GetDataPtr(), nullptr, n);

    Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    void *nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);
    Tensor masks_nb({27, n}, Dtype::Bool, device_);
    Tensor blob_indices_nb({27, n}, Dtype::Int32, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1, 3},
                           Dtype::Int64, device_);
        keys_nb[nb] = keys + dt;
    }

    hashmap_->Find(keys_nb.GetDataPtr(),
                   static_cast<iterator_t *>(nb_iterators),
                   static_cast<bool *>(masks_nb.GetDataPtr()),
                   static_cast<addr_t *>(blob_indices_nb.GetDataPtr()), n * 27);

    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor mask_nb = masks_nb[nb];
        utility::LogInfo("{}: ({}, {}, {}) => {}", nb, dx - 1, dy - 1, dz - 1,
                         mask_nb.To(Dtype::Int32).Sum({0}).Item<int>());
    }

    /// 27 * n nbs (some of them are empty), each pointing to a TSDF and a
    /// weight Tensor
    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_nb_tsdf_tl(
            static_cast<void **>(nb_iterators), 27 * n, true, {shape, shape},
            {Dtype::Float32, Dtype::Float32}, device_);
    SparseTensorList sparse_tsdf_tl(static_cast<void **>(block_iterators), n,
                                    true, {shape, shape},
                                    {Dtype::Float32, Dtype::Float32}, device_);

    Tensor dummy;
    kernel::SpecialOpEW({masks_nb}, {sparse_tsdf_tl, sparse_nb_tsdf_tl}, dummy,
                        sparse_tsdf_tl, kernel::SpecialOpCode::Check);

    MemoryManager::Free(block_iterators, device_);
    MemoryManager::Free(blob_indices, device_);
    MemoryManager::Free(nb_iterators, device_);
    return std::make_pair(sparse_nb_tsdf_tl, masks_nb);
}

PointCloud VoxelGrid::ExtractSurfacePoints() {
    int64_t n = hashmap_->Size();
    void *block_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *surf_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);

    Tensor blob_indices({n}, Dtype::Int32, device_);
    hashmap_->GetIterators(static_cast<iterator_t *>(block_iterators),
                           static_cast<addr_t *>(blob_indices.GetDataPtr()));

    Tensor keys({n, 3}, Dtype::Int64, device_);
    hashmap_->UnpackIterators(static_cast<iterator_t *>(block_iterators),
                              nullptr, keys.GetDataPtr(), nullptr, n);

    // Each voxel corresponds to ptrs to 3 vertices
    auto surface_hashmap = std::make_shared<Hashmap>(
            hashmap_->Size(),
            Dtype(Dtype::DtypeCode::Object, 3 * sizeof(int64_t), "key3d"),
            Dtype(Dtype::DtypeCode::Object,
                  3 * resolution_ * resolution_ * resolution_ * sizeof(int),
                  "vertexmap3d"),
            device_);

    Tensor masks({n}, Dtype::Bool, device_);
    surface_hashmap->Activate(
            keys.GetDataPtr(), static_cast<iterator_t *>(surf_iterators),
            static_cast<bool *>(masks.GetDataPtr()),
            static_cast<addr_t *>(blob_indices.GetDataPtr()), n);

    // Each block corresponds to 27 neighbors (at most)
    Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    Tensor masks_nb({27, n}, Dtype::Bool, device_);
    Tensor blob_indices_nb({27, n}, Dtype::Int32, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1, 3},
                           Dtype::Int64, device_);
        keys_nb[nb] = keys + dt;
    }
    hashmap_->Find(keys_nb.GetDataPtr(),
                   static_cast<iterator_t *>(nb_iterators),
                   static_cast<bool *>(masks_nb.GetDataPtr()),
                   static_cast<addr_t *>(blob_indices_nb.GetDataPtr()), n * 27);

    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor mask_nb = masks_nb[nb];
        utility::LogInfo("{}: ({}, {}, {}) => {}", nb, dx - 1, dy - 1, dz - 1,
                         mask_nb.To(Dtype::Int32).Sum({0}).Item<int>());
    }

    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};
    SparseTensorList sparse_tsdf_tl(static_cast<void **>(block_iterators), n,
                                    true, {shape, shape},
                                    {Dtype::Float32, Dtype::Float32}, device_);

    SparseTensorList sparse_surf_tl(static_cast<void **>(surf_iterators), n,
                                    true, {shape, shape, shape},
                                    {Dtype::Int32, Dtype::Int32, Dtype::Int32},
                                    device_);
    SparseTensorList sparse_nb_tsdf_tl(
            static_cast<void **>(nb_iterators), 27 * n, true, {shape, shape},
            {Dtype::Float32, Dtype::Float32}, device_);

    Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);
    Tensor output;

    utility::LogInfo("Launch");
    kernel::SpecialOpEW({voxel_size, masks_nb},
                        {sparse_tsdf_tl, sparse_nb_tsdf_tl}, output,
                        sparse_surf_tl, kernel::SpecialOpCode::ExtractSurface);

    utility::LogInfo("Free");
    MemoryManager::Free(block_iterators, device_);
    MemoryManager::Free(surf_iterators, device_);
    MemoryManager::Free(nb_iterators, device_);

    utility::LogInfo("Point");
    return PointCloud(TensorList::FromTensor(output.T()));
}

PointCloud VoxelGrid::MarchingCubes() {
    int64_t n = hashmap_->Size();
    void *tsdf_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *surf_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n, device_);
    void *tsdf_nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);
    void *surf_nb_iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * n * 27, device_);

    // Fetch active keys
    Tensor keys({n, 3}, Dtype::Int64, device_);
    Tensor blob_indices({n}, Dtype::Int32, device_);
    hashmap_->GetIterators(static_cast<iterator_t *>(tsdf_iterators),
                           static_cast<addr_t *>(blob_indices.GetDataPtr()));
    hashmap_->UnpackIterators(static_cast<iterator_t *>(tsdf_iterators),
                              nullptr, keys.GetDataPtr(), nullptr, n);

    // Activate relevant surfaces
    Tensor masks({n}, Dtype::Bool, device_);
    auto surf_hashmap = std::make_shared<Hashmap>(
            hashmap_->Size(),
            Dtype(Dtype::DtypeCode::Object, 3 * sizeof(int64_t), "key3d"),
            Dtype(Dtype::DtypeCode::Object,
                  4 * resolution_ * resolution_ * resolution_ * sizeof(int),
                  "mcmap3d"),
            device_);
    surf_hashmap->Activate(keys.GetDataPtr(),
                           static_cast<iterator_t *>(surf_iterators),
                           static_cast<bool *>(masks.GetDataPtr()),
                           static_cast<addr_t *>(blob_indices.GetDataPtr()), n);

    // Get neighbors per tsdf / surf block
    Tensor keys_nb({27, n, 3}, Dtype::Int64, device_);
    Tensor masks_nb({27, n}, Dtype::Bool, device_);
    Tensor blob_indices_nb({27, n}, Dtype::Int32, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        Tensor dt = Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1}, {1, 3},
                           Dtype::Int64, device_);
        keys_nb[nb] = keys + dt;
    }

    hashmap_->Find(keys_nb.GetDataPtr(),
                   static_cast<iterator_t *>(tsdf_nb_iterators),
                   static_cast<bool *>(masks_nb.GetDataPtr()),
                   static_cast<addr_t *>(blob_indices_nb.GetDataPtr()), n * 27);
    surf_hashmap->Find(
            keys_nb.GetDataPtr(), static_cast<iterator_t *>(surf_nb_iterators),
            static_cast<bool *>(masks_nb.GetDataPtr()),
            static_cast<addr_t *>(blob_indices_nb.GetDataPtr()), n * 27);

    SizeVector shape = SizeVector{resolution_, resolution_, resolution_};

    SparseTensorList sparse_tsdf_tl(static_cast<void **>(tsdf_iterators), n,
                                    true, {shape, shape},
                                    {Dtype::Float32, Dtype::Float32}, device_);
    SparseTensorList sparse_nb_tsdf_tl(
            static_cast<void **>(tsdf_nb_iterators), 27 * n, true,
            {shape, shape}, {Dtype::Float32, Dtype::Float32}, device_);

    SparseTensorList sparse_surf_tl(
            static_cast<void **>(surf_iterators), n, true,
            {shape, shape, shape, shape},
            {Dtype::Int32, Dtype::Int32, Dtype::Int32, Dtype::Int32}, device_);
    SparseTensorList sparse_nb_surf_tl(
            static_cast<void **>(surf_nb_iterators), 27 * n, true,
            {shape, shape, shape, shape},
            {Dtype::Int32, Dtype::Int32, Dtype::Int32, Dtype::Int32}, device_);

    Tensor voxel_size(std::vector<float>{voxel_size_}, {1}, Dtype::Float32);

    Tensor triangle_count;
    kernel::SpecialOpEW({voxel_size, masks_nb},
                        {sparse_tsdf_tl, sparse_nb_tsdf_tl}, triangle_count,
                        sparse_nb_surf_tl,
                        kernel::SpecialOpCode::MarchingCubesPass0);

    Tensor vertices;
    kernel::SpecialOpEW({voxel_size, triangle_count, masks_nb},
                        {sparse_tsdf_tl, sparse_nb_tsdf_tl}, vertices,
                        sparse_surf_tl,
                        kernel::SpecialOpCode::MarchingCubesPass1);

    Tensor triangles;
    kernel::SpecialOpEW({voxel_size, triangle_count, masks_nb},
                        {sparse_surf_tl}, triangles, sparse_nb_surf_tl,
                        kernel::SpecialOpCode::MarchingCubesPass2);
    MemoryManager::Free(tsdf_iterators, device_);
    MemoryManager::Free(tsdf_nb_iterators, device_);

    MemoryManager::Free(surf_iterators, device_);
    MemoryManager::Free(surf_nb_iterators, device_);

    auto pcd = PointCloud(TensorList::FromTensor(vertices.T().Slice(1, 0, 3)));
    pcd.SetPointNormals(TensorList::FromTensor(vertices.T().Slice(1, 3, 6)));
    pcd.SetPointAttr("triangles",
                     TensorList::FromTensor(triangles.Copy(device_)));
    return pcd;
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
