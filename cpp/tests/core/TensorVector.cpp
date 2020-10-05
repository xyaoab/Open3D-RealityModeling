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

#include "open3d/core/TensorVector.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorVectorPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorVector,
                         TensorVectorPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorVectorPermuteDevices, EmptyConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    // TensorVector allows 0-sized and scalar {} element_shape.
    for (const core::SizeVector& element_shape : std::vector<core::SizeVector>{
                 {},   // Scalar {} element_shape is fine.
                 {0},  // 0-sized element_shape is fine.
                 {1},  // This is different from {}.
                 {0, 0},
                 {0, 1},
                 {1, 0},
                 {2, 3},
         }) {
        core::TensorVector tl(element_shape, dtype, device);
        EXPECT_EQ(tl.GetElementShape(), element_shape);
        EXPECT_EQ(tl.GetDtype(), dtype);
        EXPECT_EQ(tl.GetDevice(), device);
    }

    // TensorVector does not allow negative element_shape.
    EXPECT_ANY_THROW(core::TensorVector({0, -1}, dtype, device));
    EXPECT_ANY_THROW(core::TensorVector({-1, -1}, dtype, device));
}

TEST_P(TensorVectorPermuteDevices, ConstructFromTensorVector) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0.;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1.;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2.;
    core::TensorVector tl(std::vector<core::Tensor>({t0, t1, t2}));

    // Check tensor list.
    core::SizeVector full_shape({3, 2, 3});
    EXPECT_EQ(tl.AsTensor().GetShape(), full_shape);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);

    // Values should be copied. IsClose also ensures the same dtype and device.
    EXPECT_TRUE(tl[0].AllClose(t0));
    EXPECT_TRUE(tl[1].AllClose(t1));
    EXPECT_TRUE(tl[2].AllClose(t2));
    EXPECT_FALSE(tl[0].IsSame(t0));
    EXPECT_FALSE(tl[1].IsSame(t1));
    EXPECT_FALSE(tl[2].IsSame(t2));

    // Device mismatch.
    core::Tensor t3 = core::Tensor::Ones({2, 3}, dtype, core::Device("CPU:0"));
    core::Tensor t4 = core::Tensor::Ones({2, 3}, dtype, device);
    if (t3.GetDevice() != t4.GetDevice()) {
        // This tests only fires when CUDA is available.
        EXPECT_ANY_THROW(
                core::TensorVector(std::vector<core::Tensor>({t3, t4})));
    }

    // Shape mismatch.
    core::Tensor t5 = core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor t6 = core::Tensor::Ones({2, 3}, core::Dtype::Float64, device);
    EXPECT_ANY_THROW(core::TensorVector(std::vector<core::Tensor>({t5, t6})));
}

TEST_P(TensorVectorPermuteDevices, ConstructFromTensors) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0.;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1.;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2.;
    std::vector<core::Tensor> tensors({t0, t1, t2});

    for (const core::TensorVector& tl : std::vector<core::TensorVector>({
                 core::TensorVector(tensors),
                 core::TensorVector(tensors.begin(), tensors.end()),
                 core::TensorVector({t0, t1, t2}),
         })) {
        core::SizeVector full_shape({3, 2, 3});
        EXPECT_EQ(tl.AsTensor().GetShape(), full_shape);
        EXPECT_EQ(tl.GetSize(), 3);
        EXPECT_EQ(tl.GetReservedSize(), 8);
        // Values are the same.
        EXPECT_TRUE(tl[0].AllClose(t0));
        EXPECT_TRUE(tl[1].AllClose(t1));
        EXPECT_TRUE(tl[2].AllClose(t2));
        // Tensors are copied.
        EXPECT_FALSE(tl[0].IsSame(t0));
        EXPECT_FALSE(tl[1].IsSame(t1));
        EXPECT_FALSE(tl[2].IsSame(t2));
    }

    // Device mismatch.
    core::Tensor t3 = core::Tensor::Ones({2, 3}, dtype, core::Device("CPU:0"));
    core::Tensor t4 = core::Tensor::Ones({2, 3}, dtype, device);
    if (t3.GetDevice() != t4.GetDevice()) {
        // This tests only fires when CUDA is available.
        EXPECT_ANY_THROW(
                core::TensorVector(std::vector<core::Tensor>({t3, t4})));
    }

    // Shape mismatch.
    core::Tensor t5 = core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor t6 = core::Tensor::Ones({2, 3}, core::Dtype::Float64, device);
    EXPECT_ANY_THROW(core::TensorVector(std::vector<core::Tensor>({t5, t6})));
}

TEST_P(TensorVectorPermuteDevices, FromTensor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    // Copyied tensor.
    core::TensorVector tl = core::TensorVector::FromTensor(t);
    EXPECT_EQ(tl.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);
    EXPECT_TRUE(tl.AsTensor().AllClose(t));
    EXPECT_FALSE(tl.AsTensor().IsSame(t));

    // Inplace tensor.
    core::TensorVector tl_inplace = core::TensorVector::FromTensor(t, true);
    EXPECT_EQ(tl_inplace.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl_inplace.GetSize(), 3);
    EXPECT_EQ(tl_inplace.GetReservedSize(), 3);
    EXPECT_TRUE(tl_inplace.AsTensor().AllClose(t));
    EXPECT_TRUE(tl_inplace.AsTensor().IsSame(t));
}

TEST_P(TensorVectorPermuteDevices, CopyConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    core::TensorVector tl = core::TensorVector::FromTensor(t, false);
    core::TensorVector tl_copy(tl);
    EXPECT_TRUE(tl.AsTensor().IsSame(tl_copy.AsTensor()));
}

TEST_P(TensorVectorPermuteDevices, MoveConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    auto create_tl = [&t]() {
        return core::TensorVector::FromTensor(t, /*inplace=*/true);
    };
    core::TensorVector tl(create_tl());
    EXPECT_TRUE(tl.AsTensor().IsSame(t));
}

TEST_P(TensorVectorPermuteDevices, CopyAssignmentOperator) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    // Initially tl_a and tl_b does not share the same underlying memory.
    core::TensorVector tl_a = core::TensorVector::FromTensor(t);
    core::TensorVector tl_b = core::TensorVector::FromTensor(t);
    EXPECT_TRUE(tl_a.AsTensor().AllClose(tl_b.AsTensor()));
    EXPECT_FALSE(tl_a.AsTensor().IsSame(tl_b.AsTensor()));

    // After copy assignment, the underlying memory are the same.
    tl_a = tl_b;
    EXPECT_TRUE(tl_a.AsTensor().AllClose(tl_b.AsTensor()));
    EXPECT_TRUE(tl_a.AsTensor().IsSame(tl_b.AsTensor()));

    // The is_resizable_ property will be overwritten.
    core::TensorVector tl_a_inplace = core::TensorVector::FromTensor(t, true);
    EXPECT_FALSE(tl_a_inplace.IsResizable());
    tl_a_inplace = tl_b;
    EXPECT_TRUE(tl_a_inplace.IsResizable());
}

TEST_P(TensorVectorPermuteDevices, MoveAssignmentOperator) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t_a = core::Tensor::Ones({3, 4, 5}, dtype, device);
    core::Tensor t_b = core::Tensor::Ones({3, 4, 5}, dtype, device);

    core::TensorVector tl_a =
            core::TensorVector::FromTensor(t_a, /*inplace=*/true);
    auto create_tl_b = [&t_b]() {
        return core::TensorVector::FromTensor(t_b, /*inplace=*/true);
    };
    EXPECT_FALSE(tl_a.AsTensor().IsSame(t_b));
    tl_a = create_tl_b();
    EXPECT_TRUE(tl_a.AsTensor().IsSame(t_b));
}

TEST_P(TensorVectorPermuteDevices, CopyFrom) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t_a = core::Tensor::Ones({10, 4, 5}, dtype, device);
    core::Tensor t_b = core::Tensor::Ones({0, 2, 3}, dtype, device);

    core::TensorVector tl_a = core::TensorVector::FromTensor(t_a);
    core::TensorVector tl_b = core::TensorVector::FromTensor(t_b);
    EXPECT_NE(tl_a.GetElementShape(), tl_b.GetElementShape());
    EXPECT_NE(tl_a.GetSize(), tl_b.GetSize());
    EXPECT_NE(tl_a.GetReservedSize(), tl_b.GetReservedSize());

    tl_b.CopyFrom(tl_a);
    EXPECT_EQ(tl_a.GetElementShape(), tl_b.GetElementShape());
    EXPECT_EQ(tl_a.GetSize(), tl_b.GetSize());
    EXPECT_EQ(tl_a.GetReservedSize(), tl_b.GetReservedSize());
    EXPECT_TRUE(tl_b.AsTensor().AllClose(tl_a.AsTensor()));
    EXPECT_FALSE(tl_b.AsTensor().IsSame(tl_a.AsTensor()));

    // The is_resizable_ property will be overwritten.
    core::TensorVector tl_a_inplace = core::TensorVector::FromTensor(t_a, true);
    EXPECT_FALSE(tl_a_inplace.IsResizable());
    tl_a_inplace.CopyFrom(tl_b);
    EXPECT_TRUE(tl_a_inplace.IsResizable());
}

TEST_P(TensorVectorPermuteDevices, ShallowCopyFrom) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t_a = core::Tensor::Ones({10, 4, 5}, dtype, device);
    core::Tensor t_b = core::Tensor::Ones({0, 2, 3}, dtype, device);

    core::TensorVector tl_a = core::TensorVector::FromTensor(t_a);
    core::TensorVector tl_b = core::TensorVector::FromTensor(t_b);
    EXPECT_NE(tl_a.GetElementShape(), tl_b.GetElementShape());
    EXPECT_NE(tl_a.GetSize(), tl_b.GetSize());
    EXPECT_NE(tl_a.GetReservedSize(), tl_b.GetReservedSize());

    tl_b.ShallowCopyFrom(tl_a);
    EXPECT_EQ(tl_a.GetElementShape(), tl_b.GetElementShape());
    EXPECT_EQ(tl_a.GetSize(), tl_b.GetSize());
    EXPECT_EQ(tl_a.GetReservedSize(), tl_b.GetReservedSize());
    EXPECT_TRUE(tl_b.AsTensor().AllClose(tl_a.AsTensor()));
    EXPECT_TRUE(tl_b.AsTensor().IsSame(tl_a.AsTensor()));

    // The is_resizable_ property will be overwritten.
    core::TensorVector tl_a_inplace = core::TensorVector::FromTensor(t_a, true);
    EXPECT_FALSE(tl_a_inplace.IsResizable());
    tl_a_inplace.ShallowCopyFrom(tl_b);
    EXPECT_TRUE(tl_a_inplace.IsResizable());
}

TEST_P(TensorVectorPermuteDevices, CopyBecomesResizable) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t_a = core::Tensor::Ones({10, 4, 5}, dtype, device);
    core::Tensor t_b = core::Tensor::Ones({0, 2, 3}, dtype, device);
    core::Tensor t = core::Tensor::Ones({4, 5}, dtype, device);

    core::TensorVector tl_a = core::TensorVector::FromTensor(t_a, true);
    core::TensorVector tl_b = core::TensorVector::FromTensor(t_b, true);
    core::TensorVector tl_b_backup = tl_b;
    EXPECT_FALSE(tl_a.IsResizable());
    EXPECT_FALSE(tl_b.IsResizable());
    EXPECT_FALSE(tl_b_backup.IsResizable());

    tl_b.CopyFrom(tl_a);
    tl_b.PushBack(t);
    EXPECT_TRUE(tl_b.IsResizable());
    EXPECT_FALSE(tl_b_backup.IsResizable());  // tl_b_backup is not affected.
    EXPECT_EQ(tl_b_backup.GetSize(), 0);
}

TEST_P(TensorVectorPermuteDevices, Resize) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor t = core::Tensor::Ones({3, 4, 5}, dtype, device);

    core::TensorVector tl = core::TensorVector::FromTensor(t);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);
    EXPECT_TRUE(tl.AsTensor().AllClose(t));

    tl.Resize(5);
    EXPECT_EQ(tl.GetSize(), 5);
    EXPECT_EQ(tl.GetReservedSize(), 16);
    EXPECT_TRUE(tl.AsTensor().Slice(0, 0, 3).AllClose(t));
    EXPECT_TRUE(tl.AsTensor().Slice(0, 3, 5).AllClose(
            core::Tensor::Zeros({2, 4, 5}, dtype, device)));

    tl.Resize(2);
    EXPECT_EQ(tl.GetSize(), 2);
    EXPECT_EQ(tl.GetReservedSize(), 16);
    EXPECT_TRUE(tl.AsTensor().AllClose(
            core::Tensor::Ones({2, 4, 5}, dtype, device)));

    tl = core::TensorVector::FromTensor(t, /*inplace=*/true);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 3);
    EXPECT_ANY_THROW(tl.Resize(5));

    // Inplace TensorVector does not suport resize.
    core::TensorVector tl_inplace = core::TensorVector::FromTensor(t, true);
    EXPECT_ANY_THROW(tl_inplace.Resize(2));
}

TEST_P(TensorVectorPermuteDevices, PushBack) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2;

    // Start from emtpy tensor list
    core::TensorVector tl({2, 3}, core::Dtype::Float32, device);
    EXPECT_EQ(tl.GetSize(), 0);
    EXPECT_EQ(tl.GetReservedSize(), 1);

    tl.PushBack(t0);
    EXPECT_EQ(tl.GetSize(), 1);
    EXPECT_EQ(tl.GetReservedSize(), 2);
    EXPECT_TRUE(tl[0].AllClose(t0));
    EXPECT_FALSE(tl[0].IsSame(t0));  // Values should be copied

    tl.PushBack(t1);
    EXPECT_EQ(tl.GetSize(), 2);
    EXPECT_EQ(tl.GetReservedSize(), 4);
    EXPECT_TRUE(tl[1].AllClose(t1));
    EXPECT_FALSE(tl[1].IsSame(t1));

    tl.PushBack(t2);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);
    EXPECT_TRUE(tl[2].AllClose(t2));
    EXPECT_FALSE(tl[2].IsSame(t2));

    // Inplace TensorVector does not suport push back.
    core::TensorVector tl_inplace = core::TensorVector::FromTensor(
            core::Tensor::Ones({3, 2, 3}, dtype, device), true);
    EXPECT_ANY_THROW(tl_inplace.PushBack(t0));
}

TEST_P(TensorVectorPermuteDevices, Extend) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Zeros({1, 2, 3}, dtype, device);
    core::Tensor t1 = core::Tensor::Ones({3, 2, 3}, dtype, device);
    core::TensorVector tl0 = core::TensorVector::FromTensor(t0);
    core::TensorVector tl1 = core::TensorVector::FromTensor(t1);

    tl1.Extend(tl1);
    EXPECT_EQ(tl1.GetSize(), 6);
    EXPECT_TRUE(tl1.AsTensor().AllClose(
            core::Tensor::Ones({6, 2, 3}, dtype, device)));
    tl1.Extend(tl0);
    EXPECT_EQ(tl1.GetSize(), 7);
    EXPECT_TRUE(tl1[6].AllClose(core::Tensor::Zeros({2, 3}, dtype, device)));

    // Inplace TensorVector cannot be extended.
    core::TensorVector tl0_inplace = core::TensorVector::FromTensor(t0, true);
    EXPECT_ANY_THROW(tl0_inplace.Extend(tl0));

    // Inplace TensorVector can be the extension part.
    tl1.Extend(tl0_inplace);
    EXPECT_TRUE(tl1[7].AllClose(core::Tensor::Zeros({2, 3}, dtype, device)));
}

TEST_P(TensorVectorPermuteDevices, Concatenate) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Zeros({1, 2, 3}, dtype, device);
    core::Tensor t1 = core::Tensor::Ones({3, 2, 3}, dtype, device);
    core::TensorVector tl0 = core::TensorVector::FromTensor(t0);
    core::TensorVector tl1 = core::TensorVector::FromTensor(t1);

    core::TensorVector tl2 = tl0 + tl1;
    EXPECT_EQ(tl2.GetSize(), 4);
    EXPECT_EQ(tl2.GetReservedSize(), 8);
    EXPECT_TRUE(tl2.AsTensor().Slice(0, 0, 1).AllClose(t0));
    EXPECT_TRUE(tl2.AsTensor().Slice(0, 1, 4).AllClose(t1));

    core::TensorVector tl3 = tl1 + tl0;
    EXPECT_EQ(tl3.GetSize(), 4);
    EXPECT_EQ(tl1.GetReservedSize(), 8);
    EXPECT_EQ(tl3.GetReservedSize(), 8);
    EXPECT_TRUE(tl3.AsTensor().Slice(0, 0, 3).AllClose(tl1.AsTensor()));
    EXPECT_FALSE(tl3.AsTensor().Slice(0, 0, 3).IsSame(tl1.AsTensor()));  // Copy
    EXPECT_TRUE(tl3.AsTensor().Slice(0, 3, 4).AllClose(tl0.AsTensor()));
    EXPECT_FALSE(tl3.AsTensor().Slice(0, 3, 4).IsSame(tl0.AsTensor()));  // Copy
}

TEST_P(TensorVectorPermuteDevices, SquareBracketsOperator) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t0 = core::Tensor::Ones({2, 3}, dtype, device) * 0;
    core::Tensor t1 = core::Tensor::Ones({2, 3}, dtype, device) * 1;
    core::Tensor t2 = core::Tensor::Ones({2, 3}, dtype, device) * 2;
    core::TensorVector tl({t0, t1, t2});

    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_TRUE(tl[0].AllClose(t0));
    EXPECT_TRUE(tl[1].AllClose(t1));
    EXPECT_TRUE(tl[2].AllClose(t2));
    EXPECT_TRUE(tl[-1].AllClose(t2));
    EXPECT_TRUE(tl[-2].AllClose(t1));
    EXPECT_TRUE(tl[-3].AllClose(t0));
    EXPECT_FALSE(tl[0].IsSame(t0));
    EXPECT_FALSE(tl[1].IsSame(t1));
    EXPECT_FALSE(tl[2].IsSame(t2));
    EXPECT_ANY_THROW(tl[3]);
    EXPECT_ANY_THROW(tl[-4]);

    tl[0] = t1;
    tl[1] = t2;
    tl[-1] = t0;
    EXPECT_TRUE(tl[0].AllClose(t1));
    EXPECT_TRUE(tl[1].AllClose(t2));
    EXPECT_TRUE(tl[2].AllClose(t0));
    EXPECT_FALSE(tl[0].IsSame(t1));  // Deep copy when assigned to a slice.
    EXPECT_FALSE(tl[1].IsSame(t2));
    EXPECT_FALSE(tl[2].IsSame(t0));
}

TEST_P(TensorVectorPermuteDevices, Clear) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor t = core::Tensor::Ones({10, 4, 5}, dtype, device);
    core::TensorVector tl = core::TensorVector::FromTensor(t);
    tl.Clear();
    EXPECT_EQ(tl.GetSize(), 0);
    EXPECT_EQ(tl.GetReservedSize(), 1);

    core::TensorVector tl_inplace = core::TensorVector::FromTensor(t, true);
    EXPECT_ANY_THROW(tl_inplace.Clear());
}

}  // namespace tests
}  // namespace open3d
