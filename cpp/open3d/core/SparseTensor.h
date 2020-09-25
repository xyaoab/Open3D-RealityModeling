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

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Hashmap.h"

namespace open3d {
namespace core {

class SparseTensor {
public:
    /// Constructors
    SparseTensor(const Dtype& coords_dtype,
                 const SizeVector& coords_shape,
                 const Dtype& elems_dtype,
                 const SizeVector& elems_shape,
                 const Device& device,
                 int64_t init_capacity = 10);

    SparseTensor(const Tensor& coords,
                 const Tensor& elems,
                 bool insert = false);
    ~SparseTensor();

    /// Wrappers to hashmap
    std::pair<Tensor, Tensor> InsertEntries(const Tensor& coords,
                                            const Tensor& elems);
    std::pair<Tensor, Tensor> ActivateEntries(const Tensor& coords);
    std::pair<Tensor, Tensor> FindEntries(const Tensor& coords);
    Tensor EraseEntries(const Tensor& coords);

    /// Unpack discontiguous elements to a sequence of elems,
    /// important for converting tensors to nn.ParameterList in PyTorch.
    std::vector<Tensor> GetElems(const Tensor& iterators);

protected:
    std::shared_ptr<Hashmap> hashmap_;

    /// Used to destruct after hashmap_ is deleted
    std::shared_ptr<Blob> dummy_blob_;

    Dtype coords_dtype_;
    Dtype elems_dtype_;

    SizeVector coords_shape_;
    SizeVector elems_shape_;

    Device device_;
};
}  // namespace core
}  // namespace open3d
