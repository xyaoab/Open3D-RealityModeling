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

#pragma once

#include <thrust/device_vector.h>
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/MemoryManager.h"

#include "HashmapCUDA.h"

/*
 * Default hash function:
 * It treat any kind of input as a concatenation of ints.
 */

/* Lightweight wrapper to handle host input */
/* Key supports elementary types: int, long, etc. */
/* Value supports arbitrary types in theory. */
/* std::vector<bool> is specialized: it stores only one bit per element
 * We have to use uint8_t instead to read and write masks
 * https://en.wikipedia.org/w/index.php?title=Sequence_container_(C%2B%2B)&oldid=767869909#Specialization_for_bool
 */
namespace cuda {
class Hashmap {
public:
    using MemMgr = open3d::MemoryManager;

    Hashmap(uint32_t max_keys,
            uint32_t dsize_key,
            uint32_t dsize_value,
            // Preset hash table params to estimate bucket num
            uint32_t keys_per_bucket = 10,
            float expected_occupancy_per_bucket = 0.5,
            // CUDA device
            open3d::Device device = open3d::Device("CUDA:0"));
    ~Hashmap();

    /// Thrust interface for array-like input
    /// TODO: change to raw pointers and adapt to Tensor scheduling
    /// Insert keys and values.
    /// Return (CUDA) pointers to the inserted kv pairs.
    std::pair<thrust::device_vector<iterator_t>, thrust::device_vector<uint8_t>>
    Insert(uint8_t* input_keys, uint8_t* input_values, uint32_t input_key_size);

    /// Search keys.
    /// Return (CUDA) pointers to the found kv pairs; nullptr for not found.
    /// Also returns a mask array to indicate success.
    std::pair<thrust::device_vector<iterator_t>, thrust::device_vector<uint8_t>>
    Search(uint8_t* input_keys, uint32_t input_key_size);

    /// Remove key value pairs given keys.
    /// Return mask array to indicate success or not found.
    thrust::device_vector<uint8_t> Remove(uint8_t* input_keys,
                                          uint32_t input_key_size);

    /// Assistance functions for memory profiling.
    float ComputeLoadFactor();
    std::vector<int> CountElemsPerBucket();

private:
    // Rough estimation of total keys at max capacity.
    // TODO: change it adaptively in internal implementation
    uint32_t max_keys_;
    uint32_t num_buckets_;

    uint32_t dsize_key_;
    uint32_t dsize_value_;

    // Buffer to store temporary results
    uint8_t* output_key_buffer_;
    uint8_t* output_value_buffer_;
    iterator_t* output_iterator_buffer_;
    uint8_t* output_mask_buffer_;

    std::shared_ptr<HashmapCUDA> device_hashmap_;
    open3d::Device device_;
};

Hashmap::Hashmap(uint32_t max_keys,
                 uint32_t dsize_key,
                 uint32_t dsize_value,
                 uint32_t keys_per_bucket,
                 float expected_occupancy_per_bucket,
                 open3d::Device device /* = open3d::Device("CUDA:0") */)
    : max_keys_(max_keys),
      dsize_key_(dsize_key),
      dsize_value_(dsize_value),
      device_(device),
      device_hashmap_(nullptr) {
    // Set bucket size
    uint32_t expected_keys_per_bucket =
            expected_occupancy_per_bucket * keys_per_bucket;
    num_buckets_ = (max_keys + expected_keys_per_bucket - 1) /
                   expected_keys_per_bucket;

    // Allocate memory
    output_key_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * dsize_key_, device_);
    output_value_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * dsize_value_, device_);
    output_mask_buffer_ =
            (uint8_t*)MemMgr::Malloc(max_keys_ * sizeof(uint8_t), device_);
    output_iterator_buffer_ = (iterator_t*)MemMgr::Malloc(
            max_keys_ * sizeof(iterator_t), device_);

    // Initialize internal allocator
    device_hashmap_ = std::make_shared<HashmapCUDA>(
            num_buckets_, max_keys_, dsize_key_, dsize_value_, device_);
}

Hashmap::~Hashmap() {
    MemMgr::Free(output_key_buffer_, device_);
    MemMgr::Free(output_value_buffer_, device_);
    MemMgr::Free(output_mask_buffer_, device_);
    MemMgr::Free(output_iterator_buffer_, device_);
}

std::pair<thrust::device_vector<iterator_t>, thrust::device_vector<uint8_t>>
Hashmap::Insert(uint8_t* input_keys,
                uint8_t* input_values,
                uint32_t input_keys_size) {
    // TODO: rehash and increase max_keys_
    assert(input_keys_size <= max_keys_);

    device_hashmap_->Insert(input_keys, input_values, output_iterator_buffer_,
                            output_mask_buffer_, input_keys_size);

    thrust::device_vector<iterator_t> output_iterators(
            output_iterator_buffer_, output_iterator_buffer_ + input_keys_size);
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys_size);
    return std::make_pair(output_iterators, output_masks);
}

std::pair<thrust::device_vector<iterator_t>, thrust::device_vector<uint8_t>>
Hashmap::Search(uint8_t* input_keys, uint32_t input_keys_size) {
    assert(input_keys_size <= max_keys_);

    OPEN3D_CUDA_CHECK(cudaMemset(output_mask_buffer_, 0,
                                 sizeof(uint8_t) * input_keys_size));

    device_hashmap_->Search(input_keys, output_iterator_buffer_,
                            output_mask_buffer_, input_keys_size);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_vector<iterator_t> output_iterators(
            output_iterator_buffer_, output_iterator_buffer_ + input_keys_size);
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys_size);
    return std::make_pair(output_iterators, output_masks);
}

thrust::device_vector<uint8_t> Hashmap::Remove(uint8_t* input_keys,
                                               uint32_t input_keys_size) {
    OPEN3D_CUDA_CHECK(cudaMemset(output_mask_buffer_, 0,
                                 sizeof(uint8_t) * input_keys_size));

    device_hashmap_->Remove(input_keys, output_mask_buffer_, input_keys_size);

    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys_size);
    return output_masks;
}

std::vector<int> Hashmap::CountElemsPerBucket() {
    return device_hashmap_->CountElemsPerBucket();
}

float Hashmap::ComputeLoadFactor() {
    return device_hashmap_->ComputeLoadFactor();
}
}  // namespace cuda
