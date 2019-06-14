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

#include <cstddef>
#include <string>

#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/Shape.h"
#include "Open3D/Container/TensorBuffer.h"

// TODO: move the contents of this folder to "Open3D/src"?
//       currently they are in "open3d" top namespace but under "Array" folder
namespace open3d {

// Tensor is a wrapper for TensorBuffer
// This is useful for creating a single TensorBuffer
template <typename T>
class Tensor {
public:
    Tensor(const Shape& shape, const std::string& device = "cpu")
        : shape_(shape), device_(device) {
        tensor_buffer_ = TensorBuffer<T>();
        void* ptr = MemoryManager::Allocate(sizeof(T) * shape.NumElements(),
                                            device_);
        tensor_buffer_.v_ = static_cast<T*>(ptr);
    }
    ~Tensor() { MemoryManager::Free(tensor_buffer_.v_, "cpu"); };

public:
    TensorBuffer<T> tensor_buffer_;
    Shape shape_;
    std::string device_;
};

}  // namespace open3d
