#include <fmt/format.h>

#include "open3d/Open3D.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"

using namespace open3d;
using namespace open3d::core;

template <class T, int M, int N, int A>
Tensor FromEigen(const Eigen::Matrix<T, M, N, A>& matrix) {
    Dtype dtype = Dtype::FromType<T>();
    Eigen::Matrix<T, M, N, Eigen::RowMajor> matrix_row_major = matrix;
    return Tensor(matrix_row_major.data(), {matrix.rows(), matrix.cols()},
                  dtype);
}

int main(int argc, char** argv) {
    std::string root_path = argv[1];

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);

    auto trajectory = io::CreatePinholeCameraTrajectoryFromFile(
            fmt::format("{}/trajectory.log", root_path));

    std::vector<Device> devices{Device("CUDA:0"), Device("CPU:0")};

    t::geometry::PointCloud pcd_global(Dtype::Float32, Device("CUDA:0"));

    for (int i = 0; i < 3000; ++i) {
        std::cout << i << "\n";

        /// Load image
        std::cout << "loading\n";
        std::string image_path =
                fmt::format("{}/depth/{:06d}.png", root_path, i + 1);
        std::shared_ptr<geometry::Image> im_legacy =
                io::CreateImageFromFile(image_path);
        auto depth_legacy = im_legacy->ConvertDepthToFloatImage();
        t::geometry::Image depth = t::geometry::Image::FromLegacyImage(
                *depth_legacy, Device("CUDA:0"));

        /// Unproject
        std::cout << "unprojecting\n";
        Tensor vertex_map = depth.Unproject(intrinsic);
        utility::LogInfo("{}", vertex_map.GetShape());
        Tensor pcd_map = vertex_map.View({3, 480 * 640});

        std::cout << "constructing\n";
        t::geometry::PointCloud pcd(core::TensorList::FromTensor(pcd_map.T()));

        /// Transform
        std::cout << "transforming\n";
        Eigen::Matrix4f extrinsic =
                trajectory->parameters_[i].extrinsic_.inverse().cast<float>();
        Tensor transform = FromEigen(extrinsic).Copy(Device("CUDA:0"));
        pcd.Transform(transform);

        std::cout << "downsampling\n";
        /// Downsample and append
        t::geometry::PointCloud pcd_down = pcd.VoxelDownSample(0.05);
        pcd_global.GetPoints() += pcd_down.GetPoints();
    }

    auto pcd_vis = std::make_shared<geometry::PointCloud>(
            pcd_global.ToLegacyPointCloud());
    visualization::DrawGeometries({pcd_vis});
}
