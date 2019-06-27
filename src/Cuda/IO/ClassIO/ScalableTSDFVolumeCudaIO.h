//
// Created by wei on 3/28/19.
//

#pragma once

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

namespace open3d {
namespace io {

bool WriteScalableTSDFVolumeToBIN(const std::string &filename,
                                  cuda::ScalableTSDFVolumeCuda &volume,
                                  bool use_zlib = false);
bool ReadScalableTSDFVolumeFromBIN(const std::string &filename,
                                   cuda::ScalableTSDFVolumeCuda &volume,
                                   bool use_zlib = false,
                                   int batch_size = 5000);
}  // namespace io
}  // namespace open3d
