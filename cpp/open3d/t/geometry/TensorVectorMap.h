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

#include <string>
#include <unordered_map>

#include "open3d/core/TensorVector.h"

namespace open3d {
namespace t {
namespace geometry {

/// Map of string to TensorVector. Provides helper function to maintain a
/// synchronized size (length) for the tensorvectors.
///
/// The primary key's tensorvector's size is used as the primary size and
/// primary device. Other tensorvector's size and device should be synchronized
/// according to the primary.
class TensorVectorMap
    : public std::unordered_map<std::string, core::TensorVector> {
public:
    /// Create empty TensorVectorMap and set primary key.
    TensorVectorMap(const std::string& primary_key)
        : std::unordered_map<std::string, core::TensorVector>(),
          primary_key_(primary_key) {}

    /// Create TensorVectorMap with pre-populated values.
    TensorVectorMap(const std::string& primary_key,
                    const std::unordered_map<std::string, core::TensorVector>&
                            map_keys_to_tensorvectors)
        : std::unordered_map<std::string, core::TensorVector>(),
          primary_key_(primary_key) {
        Assign(map_keys_to_tensorvectors);
    }

    /// A primary key is always required. This constructor can be marked as
    /// delete in C++, but it is needed for pybind to bind as a generic python
    /// map interface.
    TensorVectorMap() : TensorVectorMap("Undefined") {
        utility::LogError(
                "Please construct TensorVectorMap with a primary key.");
    }

    /// Clear the current map and assign new keys and values. The primary key
    /// remains unchanged. The input \p map_keys_to_tensorvectors must at least
    /// contain the primary key. Data won't be copied, tensorvectors still share
    /// the same memory as the input.
    ///
    /// \param map_keys_to_tensorvectors. The keys and values to be assigned.
    void Assign(const std::unordered_map<std::string, core::TensorVector>&
                        map_keys_to_tensorvectors);

    /// Synchronized push back, data will be copied. Before push back,
    /// IsSizeSynchronized() must be true.
    ///
    /// \param map_keys_to_tensors The keys and values to be pushed back. It
    /// must contain the same keys and each corresponding tensor must have the
    /// same dtype and device.
    void SynchronizedPushBack(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors);

    /// Returns the primary key of the tensorvectormap.
    std::string GetPrimaryKey() const { return primary_key_; }

    /// Returns true if all tensorvectors in the map have the same size.
    bool IsSizeSynchronized() const;

    /// Assert IsSizeSynchronized().
    void AssertSizeSynchronized() const;

    /// Returns true if the key exists in the map.
    /// Same as C++20's std::unordered_map::contains().
    bool Contains(const std::string& key) const { return count(key) != 0; }

private:
    /// Asserts that \p map_keys_to_tensors has the same keys as the
    /// TensorVectorMap.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor. Typically the map
    /// is used for SynchronizedPushBack.
    void AssertTensorMapSameKeys(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors) const;

    /// Asserts that all of the tensors in \p map_keys_to_tensors have the same
    /// device as the primary tensorvector.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor. Typically the map
    /// is used for SynchronizedPushBack.
    void AssertTensorMapSameDevice(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors) const;

    /// Returns the size (length) of the primary key's tensorvector.
    int64_t GetPrimarySize() const { return at(primary_key_).GetSize(); }

    /// Returns the device of the primary key's tensorvector.
    core::Device GetPrimaryDevice() const {
        return at(primary_key_).GetDevice();
    }

    /// The primary key's tensorvector's size is used as the primary size and
    /// primary device. Other tensorvector's size and device should be
    /// synchronized according to the primary.
    std::string primary_key_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
