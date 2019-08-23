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

#include <iostream>
#include <memory>

#include <Open3D/Open3D.h>
#include <Cuda/Open3DCuda.h>
#include <Cuda/IO/ClassIO/UniformTSDFVolumeCudaIO.h>

#include "Utils.h"

using namespace open3d;
using namespace open3d::utility;
using namespace open3d::io;
using namespace open3d::camera;
using namespace open3d::geometry;
using namespace open3d::visualization;


int main(int argc, char *argv[]) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    assert(argc > 2);
    std::string input_bin = argv[1];

    float voxel_length = 0.08f;
    int voxel_resolution = 64;
    float offset = -voxel_length * voxel_resolution / 2;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    extrinsics.SetTranslation(cuda::Vector3f(offset, offset, 0));

    cuda::UniformTSDFVolumeCuda tsdf_volume(voxel_resolution, voxel_length,
                                            3 * voxel_length, extrinsics);
    cuda::UniformMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor,
                                       voxel_resolution, 40000000, 80000000);

    io::ReadUniformTSDFVolumeFromBIN(input_bin, tsdf_volume);


    mesher.MarchingCubes(tsdf_volume);
    auto mesh = std::make_shared<cuda::TriangleMeshCuda>();
    *mesh = mesher.mesh();
    visualization::DrawGeometriesWithCudaModule({mesh});

    return 0;
}
