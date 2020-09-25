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

#include "open3d/core/SparseTensor.h"

namespace open3d {
namespace core {

// SparseTensor::SparseTensor(const Dtype& coords_dtype,
//                            const SizeVector& coords_shape,
//                            const Dtype& elems_dtype,
//                            const SizeVector& elems_shape,
//                            const Device& device,
//                            int64_t init_capacity)
//     : coords_dtype_(coords_dtype),
//       elems_dtype_(elems_dtype),
//       coords_shape_(coords_shape),
//       elems_shape_(elems_shape),
//       device_(device) {
//     Dtype hash_key_dtype(Dtype::DtypeCode::Object,
//                          coords_dtype_.ByteSize() *
//                          coords_shape_.NumElements(), "coords");
//     Dtype hash_val_dtype(Dtype::DtypeCode::Object,
//                          elems_dtype_.ByteSize() *
//                          elems_shape_.NumElements(), "elems");
//     hashmap_ = std::make_shared<Hashmap>(init_capacity * 2, hash_key_dtype,
//                                          hash_val_dtype, device_);
//     dummy_blob_ = std::make_shared<Blob>(4, device_);
// }

SparseTensor::SparseTensor(const Tensor& coords,
                           const Tensor& elems,
                           bool insert /* = false */) {
    // Device check
    if (coords.GetDevice().GetType() != elems.GetDevice().GetType()) {
        utility::LogError("SparseTensor::Input tensors device mismatch.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !elems.IsContiguous()) {
        utility::LogError("SparseTensor::Input tensors must be contiguous.");
    }

    // Shape check
    auto coords_full_shape = coords.GetShape();
    auto elems_full_shape = elems.GetShape();
    if (coords_full_shape.size() != 2) {
        utility::LogError("SparseTensor::Input coords shape must be (N, D).");
    }
    if (coords_full_shape[0] <= 0) {
        utility::LogError("SparseTensor::Input coords size is not positive");
    }
    if (coords_full_shape[0] != elems_full_shape[0]) {
        utility::LogError(
                "SparseTensor::Input coords and elems size mismatch.");
    }

    coords_dtype_ = coords.GetDtype();
    coords_shape_ = coords[0].GetShape();
    elems_dtype_ = elems.GetDtype();
    elems_shape_ = elems[0].GetShape();
    device_ = coords.GetDevice();

    std::cout << coords_dtype_.ByteSize() * coords_shape_.NumElements() << " "
              << elems_dtype_.ByteSize() * elems_shape_.NumElements() << "\n";

    Dtype hash_key_dtype(Dtype::DtypeCode::Object,
                         coords_dtype_.ByteSize() * coords_shape_.NumElements(),
                         "coords");
    Dtype hash_val_dtype(Dtype::DtypeCode::Object,
                         elems_dtype_.ByteSize() * elems_shape_.NumElements(),
                         "elems");
    hashmap_ = std::make_shared<Hashmap>(
            coords_full_shape[0] * 2, hash_key_dtype, hash_val_dtype, device_);
    dummy_blob_ = std::make_shared<Blob>(4, device_);

    if (insert) {
        Tensor iterators, masks;
        hashmap_->Insert(coords, elems, iterators, masks);
    }
}

SparseTensor::~SparseTensor() {}

/// Wrappers to hashmap
std::pair<Tensor, Tensor> SparseTensor::InsertEntries(const Tensor& coords,
                                                      const Tensor& elems) {
    Tensor iterators, masks;
    hashmap_->Insert(coords, elems, iterators, masks);
    return std::make_pair(iterators, masks);
}

std::pair<Tensor, Tensor> SparseTensor::ActivateEntries(const Tensor& coords) {
    Tensor iterators, masks;
    hashmap_->Activate(coords, iterators, masks);
    return std::make_pair(iterators, masks);
}

std::pair<Tensor, Tensor> SparseTensor::FindEntries(const Tensor& coords) {
    Tensor iterators, masks;
    hashmap_->Find(coords, iterators, masks);
    return std::make_pair(iterators, masks);
}

Tensor SparseTensor::EraseEntries(const Tensor& coords) {
    Tensor masks;
    hashmap_->Erase(coords, masks);
    return masks;
}

std::vector<Tensor> SparseTensor::GetElemsList(const Tensor& iterators) {
    // We assume iterators are all valid.
    int n = iterators.GetShape()[0];
    std::vector<Tensor> sparse_tensor_list;

    for (int i = 0; i < n; ++i) {
        iterator_t it = iterators[i].Item<iterator_t>();

        sparse_tensor_list.emplace_back(elems_shape_,
                                        Tensor::DefaultStrides(elems_shape_),
                                        it.second, elems_dtype_, dummy_blob_);
    }
    return sparse_tensor_list;
}
}  // namespace core
}  // namespace open3d
