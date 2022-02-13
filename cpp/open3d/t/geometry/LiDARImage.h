// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class LiDARIntrinsic
class LiDARIntrinsic {
public:
    LiDARIntrinsic(const std::string& config_npz_file,
                   const core::Device& device);

public:
    core::Tensor lidar_to_sensor_;
    core::Tensor sensor_to_lidar_;

    core::Tensor azimuth_lut_;

    core::Tensor altitude_lut_;
    core::Tensor inv_altitude_lut_;

    core::Tensor unproj_dir_lut_;
    core::Tensor unproj_offset_lut_;

    float range_scale_ = 1000.0;
    float inv_lut_resolution_ = 0.4;
    int width_ = 1024;
};

struct LiDARIntrinsicPtrs {
public:
    LiDARIntrinsicPtrs(const LiDARIntrinsic& intrinsic);

    // Unprojection LUTs
    float* dir_lut_ptr;
    float* offset_lut_ptr;

    // Projection LUTs
    float* azimuth_lut_ptr;
    float* altitude_lut_ptr;
    int64_t* inv_altitude_lut_ptr;

    // LUT params
    float azimuth_resolution;
    int64_t inv_altitude_lut_length;
    float inv_altitude_lut_resolution;

    // Other params
    int64_t height;
    int64_t width;
};

/// \class LiDARImage
///
/// \brief The Image class stores an image, while supporting further operations
/// specified with LiDAR projection.
class LiDARImage : public Image {
public:
    /// \brief Construct from a tensor. The tensor won't be copied and memory
    /// will be shared.
    ///
    /// \param tensor: Tensor of the image. The tensor must be contiguous. The
    /// tensor must be 2D (rows, cols) or 3D (rows, cols, channels).
    LiDARImage(const core::Tensor& tensor);

    /// \brief Construct from a image, using the default copy constructor.
    LiDARImage(const Image& image) : Image(image){};

    /// Return xyz-image and mask_image, with transformation
    /// Input: range image in UInt16
    std::tuple<core::Tensor, core::Tensor> Unproject(
            const LiDARIntrinsic& intrinsic,
            const core::Tensor& transformation =
                    core::Tensor::Eye(4, core::Dtype::Float64, core::Device()),
            float depth_min = 0.65,
            float detph_max = 10.0) const;

    /// Return u, v, r, mask
    static std::tuple<core::Tensor, core::Tensor, core::Tensor, core::Tensor>
    Project(const core::Tensor& xyz,
            const LiDARIntrinsic& intrinsic,
            const core::Tensor& transformation =
                    core::Tensor::Eye(4, core::Dtype::Float64, core::Device()));

    /// Currently from point cloud, could be slow.
    core::Tensor GetNormalMap(const LiDARIntrinsic& intrinsic) const;

    /// Return
    Image Visualize(const LiDARIntrinsic& intrinsic) const;
};
}  // namespace geometry
}  // namespace t
}  // namespace open3d
