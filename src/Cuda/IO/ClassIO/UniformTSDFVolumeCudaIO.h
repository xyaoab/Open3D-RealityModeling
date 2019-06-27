//
// Created by dongw1 on 6/25/19.
//

#pragma once

#include <Cuda/Integration/UniformTSDFVolumeCuda.h>

namespace open3d {
namespace io {

bool WriteUniformTSDFVolumeToBIN(const std::string &filename,
                                 cuda::UniformTSDFVolumeCuda &volume,
                                 bool use_zlib = false);
bool ReadUniformTSDFVolumeFromBIN(const std::string &filename,
                                  cuda::UniformTSDFVolumeCuda &volume,
                                  bool use_zlib = false);
}  // namespace io
}  // namespace open3d
