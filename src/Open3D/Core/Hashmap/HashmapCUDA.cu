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

void CUDAHashmap::Setup(uint32_t max_keys,
                        uint32_t dsize_key,
                        uint32_t dsize_value,
                        open3d::Device device) {
    max_keys_ = max_keys;
    dsize_key_ = dsize_key;
    dsize_value_ = dsize_value;
    device_ = device;

    // Set bucket size
    printf("%d\n", max_keys);
    const uint32_t expected_keys_per_bucket = 10;
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
    device_hashmap_ = std::make_shared<CUDAHashmapImpl>(
            num_buckets_, max_keys_, dsize_key_, dsize_value_, device_);
}

CUDAHashmap::~CUDAHashmap() {
    printf("Freed!\n");
    MemMgr::Free(output_key_buffer_, device_);
    MemMgr::Free(output_value_buffer_, device_);
    MemMgr::Free(output_mask_buffer_, device_);
    MemMgr::Free(output_iterator_buffer_, device_);
}

std::pair<iterator_t*, uint8_t*> CUDAHashmap::Insert(uint8_t* input_keys,
                                                     uint8_t* input_values,
                                                     uint32_t input_keys_size) {
    // TODO: rehash and increase max_keys_
    assert(input_keys_size <= max_keys_);

    device_hashmap_->Insert(input_keys, input_values, output_iterator_buffer_,
                            output_mask_buffer_, input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

std::pair<iterator_t*, uint8_t*> CUDAHashmap::Search(uint8_t* input_keys,
                                                     uint32_t input_keys_size) {
    assert(input_keys_size <= max_keys_);

    device_hashmap_->Search(input_keys, output_iterator_buffer_,
                            output_mask_buffer_, input_keys_size);

    return std::make_pair(output_iterator_buffer_, output_mask_buffer_);
}

uint8_t* CUDAHashmap::Remove(uint8_t* input_keys, uint32_t input_keys_size) {
    device_hashmap_->Remove(input_keys, output_mask_buffer_, input_keys_size);

    return output_mask_buffer_;
}

}  // namespace cuda
