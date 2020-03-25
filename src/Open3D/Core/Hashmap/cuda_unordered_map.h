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
#include "slab_hash.h"
/*
 * Default hash function:
 * It treat any kind of input as a concatenation of ints.
 */
template <typename Key>
struct hash {
    __device__ __host__ uint64_t operator()(const Key& key) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        const int chunks = sizeof(Key) / sizeof(int);
        for (size_t i = 0; i < chunks; ++i) {
            hash ^= ((int32_t*)(&key))[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }
};

/* Lightweight wrapper to handle host input */
/* Key supports elementary types: int, long, etc. */
/* Value supports arbitrary types in theory. */
/* std::vector<bool> is specialized: it stores only one bit per element
 * We have to use uint8_t instead to read and write masks
 * https://en.wikipedia.org/w/index.php?title=Sequence_container_(C%2B%2B)&oldid=767869909#Specialization_for_bool
 */
namespace cuda {
template <typename Key,
          typename Value,
          typename Hash = hash<Key>,
          class Alloc = open3d::MemoryManager>
class unordered_map {
public:
    unordered_map(uint32_t max_keys,
                  /* Preset hash table params to estimate bucket num */
                  uint32_t keys_per_bucket = 10,
                  float expected_occupancy_per_bucket = 0.5,
                  /* CUDA device */
                  open3d::Device device = open3d::Device("CUDA:0"));
    ~unordered_map();

    /* Detailed output */
    std::pair<thrust::device_vector<_Iterator<Key, Value>>,
              thrust::device_vector<uint8_t>>
    Insert(thrust::device_vector<Key>& input_keys,
           thrust::device_vector<Value>& input_values);

    std::pair<thrust::device_vector<_Iterator<Key, Value>>,
              thrust::device_vector<uint8_t>>
    Search(thrust::device_vector<Key>& input_keys);

    thrust::device_vector<uint8_t> Remove(
            thrust::device_vector<Key>& input_keys);

    /* Assistance functions */
    float ComputeLoadFactor();
    std::vector<int> CountElemsPerBucket();

private:
    uint32_t max_keys_;
    uint32_t num_buckets_;

    /* Buffer for input cpu data (e.g. from std::vector) */
    Key* input_key_buffer_;
    Value* input_value_buffer_;
    Key* output_key_buffer_;
    Value* output_value_buffer_;
    _Iterator<Key, Value>* output_iterator_buffer_;
    uint8_t* output_mask_buffer_;

    std::shared_ptr<SlabHash<Key, Value, Hash, Alloc>> slab_hash_;
    open3d::Device device_;
};

template <typename Key, typename Value, typename Hash, class Alloc>
unordered_map<Key, Value, Hash, Alloc>::unordered_map(
        uint32_t max_keys,
        uint32_t keys_per_bucket,
        float expected_occupancy_per_bucket,
        open3d::Device device /* = open3d::Device("CUDA:0") */)
    : max_keys_(max_keys), device_(device), slab_hash_(nullptr) {
    /* Set bucket size */
    uint32_t expected_keys_per_bucket =
            expected_occupancy_per_bucket * keys_per_bucket;
    num_buckets_ = (max_keys + expected_keys_per_bucket - 1) /
                   expected_keys_per_bucket;

    // allocating key, value arrays to buffer input and output:
    input_key_buffer_ =
            static_cast<Key*>(Alloc::Malloc(max_keys_ * sizeof(Key), device_));
    input_value_buffer_ = static_cast<Value*>(
            Alloc::Malloc(max_keys_ * sizeof(Value), device_));
    output_key_buffer_ =
            static_cast<Key*>(Alloc::Malloc(max_keys_ * sizeof(Key), device_));
    output_value_buffer_ = static_cast<Value*>(
            Alloc::Malloc(max_keys_ * sizeof(Value), device_));
    output_mask_buffer_ = static_cast<uint8_t*>(
            Alloc::Malloc(max_keys_ * sizeof(uint8_t), device_));
    output_iterator_buffer_ = static_cast<_Iterator<Key, Value>*>(
            Alloc::Malloc(max_keys_ * sizeof(_Iterator<Key, Value>), device_));

    // allocate an initialize the allocator:
    slab_hash_ = std::make_shared<SlabHash<Key, Value, Hash, Alloc>>(
            num_buckets_, max_keys_, device_);
}

template <typename Key, typename Value, typename Hash, class Alloc>
unordered_map<Key, Value, Hash, Alloc>::~unordered_map() {
    Alloc::Free(input_key_buffer_, device_);
    Alloc::Free(input_value_buffer_, device_);

    Alloc::Free(output_key_buffer_, device_);
    Alloc::Free(output_value_buffer_, device_);
    Alloc::Free(output_mask_buffer_, device_);
    Alloc::Free(output_iterator_buffer_, device_);
}

template <typename Key, typename Value, typename Hash, class Alloc>
std::pair<thrust::device_vector<_Iterator<Key, Value>>,
          thrust::device_vector<uint8_t>>
unordered_map<Key, Value, Hash, Alloc>::Insert(
        thrust::device_vector<Key>& input_keys,
        thrust::device_vector<Value>& input_values) {
    assert(input_values.size() == input_keys.size());
    assert(input_keys.size() <= max_keys_);

    slab_hash_->Insert(thrust::raw_pointer_cast(input_keys.data()),
                       thrust::raw_pointer_cast(input_values.data()),
                       output_iterator_buffer_, output_mask_buffer_,
                       input_keys.size());

    thrust::device_vector<_Iterator<Key, Value>> output_iterators(
            output_iterator_buffer_,
            output_iterator_buffer_ + input_keys.size());
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return std::make_pair(output_iterators, output_masks);
}

template <typename Key, typename Value, typename Hash, class Alloc>
thrust::device_vector<uint8_t> unordered_map<Key, Value, Hash, Alloc>::Remove(
        thrust::device_vector<Key>& input_keys) {
    OPEN3D_CUDA_CHECK(cudaMemset(output_mask_buffer_, 0,
                                 sizeof(uint8_t) * input_keys.size()));

    slab_hash_->Remove(thrust::raw_pointer_cast(input_keys.data()),
                       output_mask_buffer_, input_keys.size());

    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return output_masks;
}

template <typename Key, typename Value, typename Hash, class Alloc>
std::pair<thrust::device_vector<_Iterator<Key, Value>>,
          thrust::device_vector<uint8_t>>
unordered_map<Key, Value, Hash, Alloc>::Search(
        thrust::device_vector<Key>& input_keys) {
    assert(input_keys.size() <= max_keys_);

    OPEN3D_CUDA_CHECK(cudaMemset(output_mask_buffer_, 0,
                                 sizeof(uint8_t) * input_keys.size()));

    slab_hash_->Search(thrust::raw_pointer_cast(input_keys.data()),
                       output_iterator_buffer_, output_mask_buffer_,
                       input_keys.size());
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_vector<_Iterator<Key, Value>> output_iterators(
            output_iterator_buffer_,
            output_iterator_buffer_ + input_keys.size());
    thrust::device_vector<uint8_t> output_masks(
            output_mask_buffer_, output_mask_buffer_ + input_keys.size());
    return std::make_pair(output_iterators, output_masks);
}

template <typename Key, typename Value, typename Hash, class Alloc>
std::vector<int> unordered_map<Key, Value, Hash, Alloc>::CountElemsPerBucket() {
    return slab_hash_->CountElemsPerBucket();
}

template <typename Key, typename Value, typename Hash, class Alloc>
float unordered_map<Key, Value, Hash, Alloc>::ComputeLoadFactor() {
    return slab_hash_->ComputeLoadFactor();
}
}  // namespace cuda
