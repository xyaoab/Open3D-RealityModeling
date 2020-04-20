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

#include "Open3D/Core/Hashmap/TensorHash.h"

using namespace open3d;

int main() {
    Device device("CUDA:0");
    Tensor insert_coords(std::vector<float>({0, 0, 1, 1, 2, 2, 3, 3, 4, 4}),
                         {5, 2}, Dtype::Float32, device);
    Tensor query_coords(std::vector<float>({0, 0, 3, 3, 1, 1, 4, 4, 8, 8}),
                        {5, 2}, Dtype::Float32, device);
    Tensor indices(std::vector<int64_t>({0, 1, 2, 3, 4}), {5}, Dtype::Int64,
                   device);

    auto tensor_hash = CreateTensorHash(insert_coords, indices);
    auto results = tensor_hash->Query(query_coords);

    /// IndexTensor [0 3 1 4 0]
    /// Tensor[shape={5}, stride={1}, Int64, CUDA:0, 0x7fce3de01800]
    utility::LogInfo("IndexTensor {}", results.first.ToString());

    /// MaskTensor [1 1 1 1 0]
    /// Tensor[shape={5}, stride={1}, UInt8, CUDA:0, 0x7fce3de01600]
    utility::LogInfo("MaskTensor {}", results.second.ToString());
}