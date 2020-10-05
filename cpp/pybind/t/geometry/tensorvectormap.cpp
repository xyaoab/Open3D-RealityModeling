// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TensorVectorMap.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_tensorvectormap(py::module& m) {
    // Bind to the generic dictionary interface such that it works the same as a
    // regular dictionay in Python, except that types are enforced. Supported
    // functions include `__bool__`, `__iter__`, `items`, `__getitem__`,
    // `__contains__`, `__delitem__`, `__len__` and map assignment.
    auto tlm = py::bind_map<TensorVectorMap>(m, "TensorVectorMap",
                                             "Map of TensorVector by string.");

    // Constructors.
    tlm.def(py::init<const std::string&>(), "primary_key"_a)
            .def(py::init<const std::string&,
                          const std::unordered_map<std::string,
                                                   core::TensorVector>&>(),
                 "primary_key"_a, "map_keys_to_tensorvectors"_a);

    // Member functions. Some C++ functions are ignored since the
    // functionalities are already covered in the generic dictionary interface.
    tlm.def("synchronized_push_back", &TensorVectorMap::SynchronizedPushBack,
            "map_keys_to_tensors"_a)
            .def("get_primary_key", &TensorVectorMap::GetPrimaryKey)
            .def("is_size_synchronized", &TensorVectorMap::IsSizeSynchronized)
            .def("assert_size_synchronized",
                 &TensorVectorMap::AssertSizeSynchronized);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
