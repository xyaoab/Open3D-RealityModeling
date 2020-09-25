#include "open3d/core/SparseTensor.h"

using namespace open3d;
int main() {
    core::Device device("CUDA:0");
    std::vector<int64_t> coords_data = {1, 2, 3, 4, 5, 6};
    core::Tensor coords(coords_data, {3, 2}, core::Dtype::Int64, device);
    std::vector<float> elems_data = {1.0, 2.0, 3.0, 2, 3, 4, 3, 4, 5,
                                     1.0, 2.0, 3.0, 2, 3, 4, 3, 4, 5};
    core::Tensor elems(elems_data, {3, 3, 2}, core::Dtype::Float32, device);

    core::SparseTensor sp_tensor(coords, elems);

    core::Tensor iterators, masks;
    std::tie(iterators, masks) = sp_tensor.InsertEntries(coords, elems);
    core::Tensor valid_iterators = iterators.IndexGet({masks});

    std::vector<core::Tensor> sp_tensor_elems =
            sp_tensor.GetElemsList(valid_iterators);

    for (auto &sp_tensor_elem : sp_tensor_elems) {
        std::cout << sp_tensor_elem.ToString() << "\n";
    }

    sp_tensor;
}
