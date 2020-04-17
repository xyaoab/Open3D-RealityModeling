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

    auto hashmap = IndexTensorCoords(insert_coords, indices);
    auto results = QueryTensorCoords(hashmap, query_coords);
    utility::LogInfo("{} {}", results.first.ToString(),
                     results.second.ToString());
}