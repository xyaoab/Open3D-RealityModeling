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

#include "Open3D/Core/Hashmap/Hashmap.cuh"
#include "Open3D/Core/Tensor.h"

namespace open3d {
std::shared_ptr<Hashmap<DefaultHash>> IndexTensorCoords(Tensor coords,
                                                        Tensor indices) {
    if (!coords.IsContiguous() || !indices.IsContiguous()) {
        utility::LogError("TensorHashCUDA::Input tensors must be contiguous.");
    }

    auto coords_shape = coords.GetShape();
    auto indices_shape = indices.GetShape();
    if (coords_shape.size() != 2 || indices_shape.size() != 1) {
        utility::LogError(
                "TensorHashCUDA::Input coords tensor must be N x Dim and "
                "indices must be N x 1");
    }
    if (coords_shape[0] != indices_shape[0]) {
        utility::LogError(
                "TensorHashCUDA::Input coords and indices shape mismatch.");
    }
    size_t N = coords_shape[0];

    if (indices.GetDtype() != Dtype::Int64 &&
        indices.GetDtype() != Dtype::Int32) {
        utility::LogError(
                "TensorHashCUDA::Input indices tensor must be Integers.");
    }

    size_t key_size = DtypeUtil::ByteSize(coords.GetDtype()) * coords_shape[1];
    size_t value_size = DtypeUtil::ByteSize(indices.GetDtype());
    utility::LogInfo("{} {}", key_size, value_size);

    auto hashmap = CreateHashmap<DefaultHash>(N * 2, key_size, value_size,
                                              coords.GetDevice());
    hashmap->Insert(static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()),
                    static_cast<uint8_t*>(indices.GetBlob()->GetDataPtr()), N);
    return hashmap;
}

__global__ void DispatchIteratorsKernel(iterator_t* iterators,
                                        uint8_t* values,
                                        uint8_t* masks,
                                        size_t key_size,
                                        size_t value_size,
                                        size_t N) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N && masks[tid]) {
        uint8_t* iterator = iterators[tid];
        uint8_t* src_value_ptr = iterator + key_size;
        uint8_t* dst_value_ptr = values + value_size * tid;
        for (int i = 0; i < value_size; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}

std::pair<Tensor, Tensor> QueryTensorCoords(
        std::shared_ptr<Hashmap<DefaultHash>> hashmap, Tensor coords) {
    // TODO: sanity check
    int64_t N = coords.GetShape()[0];
    auto result = hashmap->Search(
            static_cast<uint8_t*>(coords.GetBlob()->GetDataPtr()), N);

    auto iterators_buf = result.first;
    auto masks_buf = result.second;

    // Copy mask (avoid duplicate data)
    auto blob = std::make_shared<Blob>(hashmap->device_,
                                       static_cast<void*>(masks_buf),
                                       [](void* dummy) -> void {});
    auto mask_tensor =
            Tensor(SizeVector({N}), SizeVector({1}),
                   static_cast<void*>(masks_buf), Dtype::UInt8, blob);
    auto ret_mask_tensor = mask_tensor.Copy(hashmap->device_);

    // Dispatch values
    const size_t num_threads = 32;
    const size_t num_blocks = (N + num_threads - 1) / num_threads;

    // TODO: store value Dtype in hashmap wrapper
    auto ret_value_tensor =
            Tensor(SizeVector({N}), Dtype::Int64, hashmap->device_);

    size_t key_size =
            DtypeUtil::ByteSize(coords.GetDtype()) * coords.GetShape()[1];
    size_t value_size = DtypeUtil::ByteSize(Dtype::Int64);

    DispatchIteratorsKernel<<<num_blocks, num_threads>>>(
            iterators_buf,
            static_cast<uint8_t*>(ret_value_tensor.GetBlob()->GetDataPtr()),
            masks_buf, key_size, value_size, N);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    return std::make_pair(ret_value_tensor, ret_mask_tensor);
}
}  // namespace open3d
