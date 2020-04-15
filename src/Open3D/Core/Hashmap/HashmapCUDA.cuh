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

#include "Hashmap.h"
#include "HashmapCUDAImpl.cuh"

namespace open3d {

/// Default hash function for all types
uint64_t OPEN3D_HOST_DEVICE default_hash_fn(uint8_t* key_ptr,
                                            uint32_t key_size) {
    uint64_t hash = UINT64_C(14695981039346656037);

    const int chunks = key_size / sizeof(int);
    int32_t* cast_key_ptr = (int32_t*)(key_ptr);
    for (size_t i = 0; i < chunks; ++i) {
        hash ^= cast_key_ptr[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

hash_t __device__ default_hash_fn_ptr = default_hash_fn;

struct DefaultHash {
    uint64_t OPEN3D_HOST_DEVICE operator()(uint8_t* key_ptr,
                                           uint32_t key_size) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = key_size / sizeof(int);
        int32_t* cast_key_ptr = (int32_t*)(key_ptr);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

template <typename Hash>
CUDAHashmap<Hash>::CUDAHashmap(uint32_t max_keys,
                               uint32_t dsize_key,
                               uint32_t dsize_value,
                               open3d::Device device,
                               hash_t hash_fn_ptr)
    : Hashmap(max_keys, dsize_key, dsize_value, device, hash_fn_ptr) {
    const uint32_t expected_keys_per_bucket = 10;
    num_buckets_ = (max_keys + expected_keys_per_bucket - 1) /
                   expected_keys_per_bucket;

    output_key_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * dsize_key_, device_);
    output_value_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * dsize_value_, device_);
    output_mask_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * sizeof(uint8_t), device_);
    output_iterator_buffer_ = (iterator_t*)MemMgr::Malloc(
            max_keys_ * sizeof(iterator_t), device_);

    OPEN3D_CUDA_CHECK(cudaMemcpyFromSymbol(&hash_fn_ptr, default_hash_fn_ptr,
                                           sizeof(hash_t)));
    cuda_hashmap_impl_ = std::make_shared<CUDAHashmapImpl>(
            num_buckets_, max_keys_, dsize_key_, dsize_value_, hash_fn_ptr,
            device_);
}

template <typename Hash>
CUDAHashmap<Hash>::~CUDAHashmap() {
    MemMgr::Free(output_key_buffer_, device_);
    MemMgr::Free(output_value_buffer_, device_);
    MemMgr::Free(output_mask_buffer_, device_);
    MemMgr::Free(output_iterator_buffer_, device_);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CUDAHashmap<Hash>::Insert(
        uint8_t* input_keys, uint8_t* input_values, uint32_t input_keys_size) {
    // TODO: rehash and increase max_keys_
    if (input_keys_size > max_keys_) {
        utility::LogError(
                "CUDAHashmap::Insert: number of input keys {} larger than "
                "reserved number of keys {}",
                input_keys_size, max_keys_);
    }

    cuda_hashmap_impl_->Insert(input_keys, input_values,
                               output_iterator_buffer_, output_mask_buffer_,
                               input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

template <typename Hash>
std::pair<iterator_t*, uint8_t*> CUDAHashmap<Hash>::Search(
        uint8_t* input_keys, uint32_t input_keys_size) {
    if (input_keys_size > max_keys_) {
        utility::LogError(
                "CUDAHashmap::Search: number of input keys {} larger than "
                "reserved number of keys {}",
                input_keys_size, max_keys_);
    }

    cuda_hashmap_impl_->Search(input_keys, output_iterator_buffer_,
                               output_mask_buffer_, input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

template <typename Hash>
uint8_t* CUDAHashmap<Hash>::Remove(uint8_t* input_keys,
                                   uint32_t input_keys_size) {
    cuda_hashmap_impl_->Remove(input_keys, output_mask_buffer_,
                               input_keys_size);

    return output_mask_buffer_;
}

}  // namespace open3d
