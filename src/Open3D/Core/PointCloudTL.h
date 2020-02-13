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

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/Broadcast.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Core/TensorList.h"
namespace open3d {

class PointCloudTL {
public:
    PointCloudTL() {
        point_dict_.emplace("points",
                            TensorList(SizeVector({3}), Dtype::Float32));
    };

    /// Construct from default points
    /// points_tensor: (N, 3)
    PointCloudTL(const Tensor &points_tensor) {
        auto shape = points_tensor.GetShape();
        if (shape[1] != 3) {
            utility::LogError(
                    "PointCloud must be constructed from (N, 3) points.");
        }
        point_dict_.emplace("points", TensorList(points_tensor));
    }

    /// Construct from points and various other properties
    PointCloudTL(const std::unordered_map<std::string, Tensor> &point_dict) {
        // "point" TensorList
        auto it = point_dict.find("points");
        if (it == point_dict.end()) {
            utility::LogError("PointCloud must include key \"points\".");
        }

        auto shape = it->second.GetShape();
        if (shape[1] != 3) {
            utility::LogError(
                    "PointCloud must be constructed from (N, 3) points.");
        }

        for (auto kv : point_dict) {
            point_dict_.emplace(kv.first, kv.second);
        }
    }

    TensorList &operator[](const std::string &key) {
        auto it = point_dict_.find(key);
        if (it == point_dict_.end()) {
            utility::LogError("Unknown key {} in PointCloud dictionary.", key);
        }

        return it->second;
    }

    void SyncPushBack(
            const std::unordered_map<std::string, Tensor> &point_struct) {
        // Check if "point"" exists
        auto it = point_struct.find("points");
        if (it == point_struct.end()) {
            utility::LogError("Point must include key \"points\".");
        }

        // Lazy size check and push back
        auto size = point_dict_.find("points")->second.GetSize();
        for (auto kv : point_struct) {
            // Check existance of key in point_dict
            auto it = point_dict_.find(kv.first);
            if (it == point_dict_.end()) {
                utility::LogError("Unknown key {} in PointCloud dictionary.",
                                  kv.first);
            }

            // Check size consistency
            auto size_it = it->second.GetSize();
            if (size_it != size) {
                utility::LogError("Size mismatch ({}, {}) between ({}, {}).",
                                  "points", size, kv.first, size_it);
            }
            it->second.PushBack(kv.second);
        }
    }

private:
    std::unordered_map<std::string, TensorList> point_dict_;
};
}  // namespace open3d
