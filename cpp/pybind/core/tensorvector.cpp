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

#include <vector>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorVector.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

void pybind_core_tensorvector(py::module& m) {
    py::class_<TensorVector> tensorvector(
            m, "TensorVector",
            "A TensorVector is an extendable tensor at the 0-th dimension.");

    // Constructors.
    tensorvector.def(py::init([](const SizeVector& element_shape,
                                 const Dtype& dtype, const Device& device) {
                         return new TensorVector(element_shape, dtype, device);
                     }),
                     "element_shape"_a, "dtype"_a, "device"_a);
    tensorvector.def(py::init([](const std::vector<Tensor>& tensors) {
                         return new TensorVector(tensors);
                     }),
                     "tensors"_a);
    tensorvector.def(py::init([](const TensorVector& other) {
                         return new TensorVector(other);
                     }),
                     "other"_a);

    // Factory function.
    tensorvector.def_static("from_tensor", &TensorVector::FromTensor,
                            "tensor"_a, "inplace"_a = false);

    // Copiers.
    tensorvector.def("shallow_copy_from", &TensorVector::ShallowCopyFrom);
    tensorvector.def("copy_from", &TensorVector::CopyFrom);
    tensorvector.def("copy", &TensorVector::Copy);

    // Accessors.
    tensorvector.def("__getitem__",
                     [](TensorVector& tl, int64_t index) { return tl[index]; });
    tensorvector.def("__setitem__",
                     [](TensorVector& tl, int64_t index, const Tensor& value) {
                         tl[index] = value;
                     });
    tensorvector.def("as_tensor",
                     [](const TensorVector& tl) { return tl.AsTensor(); });
    tensorvector.def("__repr__",
                     [](const TensorVector& tl) { return tl.ToString(); });
    tensorvector.def("__str__",
                     [](const TensorVector& tl) { return tl.ToString(); });

    // Manipulations.
    tensorvector.def("push_back", &TensorVector::PushBack);
    tensorvector.def("resize", &TensorVector::Resize);
    tensorvector.def("extend", &TensorVector::Extend);
    tensorvector.def("__iadd__", &TensorVector::operator+=);
    tensorvector.def("__add__", &TensorVector::operator+);
    tensorvector.def_static("concat", &TensorVector::Concatenate);

    // Python list properties.
    // TODO: make TensorVector behave more like regular python list, see
    // std_bind.h.
    tensorvector.def("__len__", &TensorVector::GetSize);

    // Properties.
    tensorvector.def_property_readonly("size", &TensorVector::GetSize);
    tensorvector.def_property_readonly("element_shape",
                                       &TensorVector::GetElementShape);
    tensorvector.def_property_readonly("dtype", &TensorVector::GetDtype);
    tensorvector.def_property_readonly("device", &TensorVector::GetDevice);
    tensorvector.def_property_readonly("is_resizable",
                                       &TensorVector::IsResizable);
}

}  // namespace core
}  // namespace open3d
