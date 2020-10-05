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

#include "open3d/t/geometry/TriangleMesh.h"

#include "core/CoreTest.h"
#include "open3d/core/TensorVector.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TriangleMeshPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TriangleMesh,
        TriangleMeshPermuteDevicePairs,
        testing::ValuesIn(TriangleMeshPermuteDevicePairs::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    t::geometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              t::geometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default dtypes.
    EXPECT_EQ(mesh.GetVertices().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetVertices().GetDtype(), core::Dtype::Float32);
    EXPECT_EQ(mesh.GetTriangles().GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(mesh.GetTriangles().GetDtype(), core::Dtype::Int64);
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromVertices) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor vertices =
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device);
    core::Tensor single_vertex =
            core::Tensor::Ones({3}, core::Dtype::Float32, device);

    core::Tensor triangles =
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device);
    core::Tensor single_triangle =
            core::Tensor::Ones({3}, core::Dtype::Int64, device);

    // Copied, push back okay.
    t::geometry::TriangleMesh mesh(
            core::TensorVector::FromTensor(vertices, /*inplace=*/false),
            core::TensorVector::FromTensor(triangles, /*inplace=*/false));

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_EQ(mesh.GetVertices().GetSize(), 10);
    mesh.GetVertices().PushBack(single_vertex);
    EXPECT_EQ(mesh.GetVertices().GetSize(), 11);

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 10);
    mesh.GetTriangles().PushBack(single_triangle);
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 11);

    // Inplace tensorvector: cannot push_back.
    mesh = t::geometry::TriangleMesh(
            core::TensorVector::FromTensor(vertices, /*inplace=*/true),
            core::TensorVector::FromTensor(triangles, /*inplace=*/true));

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_EQ(mesh.GetVertices().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetVertices().PushBack(single_vertex));

    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 10);
    EXPECT_ANY_THROW(mesh.GetTriangles().PushBack(single_triangle));
}

TEST_P(TriangleMeshPermuteDevices, SynchronizedPushBack) {
    core::Device device = GetParam();

    // Create trianglemesh.
    core::TensorVector vertices = core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    core::TensorVector vertex_colors = core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device) * 0.5);
    core::TensorVector triangles = core::TensorVector::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Int64, device));
    core::TensorVector triangle_normals = core::TensorVector::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device) * 0.5);

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetTriangleNormals(triangle_normals);

    // Exception cases are tested in TensorVectorMap::SynchronizedPushBack().
    std::unordered_map<std::string, core::Tensor> vertex_struct;
    EXPECT_EQ(mesh.GetVertices().GetSize(), 10);
    EXPECT_EQ(mesh.GetVertexColors().GetSize(), 10);
    mesh.VertexSynchronizedPushBack({
            {"vertices", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
            {"colors", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetVertices().GetSize(), 11);
    EXPECT_EQ(mesh.GetVertexColors().GetSize(), 11);

    std::unordered_map<std::string, core::Tensor> triangle_struct;
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 5);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 5);
    mesh.TriangleSynchronizedPushBack({
            {"triangles", core::Tensor::Ones({3}, core::Dtype::Int64, device)},
            {"normals", core::Tensor::Ones({3}, core::Dtype::Float32, device)},
    });
    EXPECT_EQ(mesh.GetTriangles().GetSize(), 6);
    EXPECT_EQ(mesh.GetTriangleNormals().GetSize(), 6);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    auto vertices = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device));
    auto vertex_colors = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto vertex_labels = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    auto triangles = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device));
    auto triangle_normals = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2);
    auto triangle_labels = core::TensorVector::FromTensor(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3);

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetVertexAttr("labels", vertex_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetVertices().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.GetVertexColors().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetVertexAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetVertexNormals());

    EXPECT_TRUE(mesh.GetTriangles().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AsTensor().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::TensorVector& tl = mesh.GetVertices();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorVector& tl = mesh.GetVertexColors();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorVector& tl = mesh.GetVertexAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::TensorVector& tl = mesh.GetVertexNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::TensorVector& tl = mesh.GetTriangles();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorVector& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::TensorVector& tl =
                            mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // is device is a CUDA device.
    t::geometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                   device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::TensorVector cpu_vertices = core::TensorVector::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device));
        core::TensorVector cpu_colors = core::TensorVector::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                2);
        core::TensorVector cpu_labels = core::TensorVector::FromTensor(
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                3);

        EXPECT_ANY_THROW(mesh.SetVertices(cpu_vertices));
        EXPECT_ANY_THROW(mesh.SetVertexColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetVertexAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(core::Dtype::Float32, core::Dtype::Float64,
                                   device);
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasVertexAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetVertices(core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasVertices());
    mesh.SetTriangles(core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.HasTriangles());

    // Different size.
    mesh.SetVertexColors(core::TensorVector::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::TensorVector::FromTensor(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device)));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetVertexColors(core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::TensorVector::FromTensor(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

}  // namespace tests
}  // namespace open3d
