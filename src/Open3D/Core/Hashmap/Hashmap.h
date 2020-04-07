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
#include <cstdint>
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Hashmap/Consts.h"
#include "Open3D/Core/MemoryManager.h"
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
            open3d::Device device)
        : max_keys_(max_keys),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device){};

    virtual std::pair<iterator_t*, uint8_t*> Insert(
            uint8_t* input_keys,
            uint8_t* input_values,
            uint32_t input_key_size) = 0;

    virtual std::pair<iterator_t*, uint8_t*> Search(
            uint8_t* input_keys, uint32_t input_key_size) = 0;

    virtual uint8_t* Remove(uint8_t* input_keys, uint32_t input_key_size) = 0;

protected:
    uint32_t max_keys_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

    open3d::Device device_;
};

class CUDAHashmapImpl;
class CUDAHashmap : public Hashmap {
public:
    ~CUDAHashmap();

    CUDAHashmap(uint32_t max_keys,
                uint32_t dsize_key,
                uint32_t dsize_value,
                open3d::Device device);

    std::pair<iterator_t*, uint8_t*> Insert(uint8_t* input_keys,
                                            uint8_t* input_values,
                                            uint32_t input_key_size);

    std::pair<iterator_t*, uint8_t*> Search(uint8_t* input_keys,
                                            uint32_t input_key_size);

    uint8_t* Remove(uint8_t* input_keys, uint32_t input_key_size);

protected:
    uint32_t num_buckets_;

    // Buffer to store temporary results
    uint8_t* output_key_buffer_;
    uint8_t* output_value_buffer_;
    iterator_t* output_iterator_buffer_;
    uint8_t* output_mask_buffer_;

    std::shared_ptr<CUDAHashmapImpl> device_hashmap_;
};

std::shared_ptr<Hashmap> CreateHashmap(uint32_t max_keys,
                                       uint32_t dsize_key,
                                       uint32_t dsize_value,
                                       open3d::Device device);
}  // namespace cuda
