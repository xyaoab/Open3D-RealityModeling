//
// Created by wei on 3/28/19.
//

#include "UniformTSDFVolumeCudaIO.h"
#include "ZlibIO.h"

namespace open3d {
namespace io {

bool WriteUniformTSDFVolumeToBIN(const std::string &filename,
                                 cuda::UniformTSDFVolumeCuda &volume,
                                 bool use_zlib) {
    utility::LogInfo("Writing volume...\n");

    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<cuda::Vector3b> color;
    std::tie(tsdf, weight, color) = volume.DownloadVolume();

    FILE *fid = fopen(filename.c_str(), "wb");
    if (fid == NULL) {
        utility::LogWarning("Write BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    /** metadata **/
    if (fwrite(&volume.N_, sizeof(int), 1, fid) < 1) {
        utility::LogWarning(
                "Write BIN failed: unable to write volume resolution\n");
        return false;
    }
    if (fwrite(&volume.voxel_length_, sizeof(float), 1, fid) < 1) {
        utility::LogWarning(
                "Write BIN failed: unable to write voxel size\n");
        return false;
    }
    auto grid_to_world = volume.transform_volume_to_world_.ToEigen();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float vij = float(grid_to_world(i, j));
            if (fwrite(&vij, sizeof(float), 1, fid) < 1) {
                utility::LogWarning(
                        "Write BIN failed: unable to write transform[%d, %d]\n",
                        i, j);
                return false;
            }
        }
    }

    /** volume **/
    std::vector<uchar> compressed_buf;
    if (use_zlib) {
        compressed_buf.resize(/* provide some redundancy */
                              volume.N_ * volume.N_ * volume.N_ *
                              sizeof(float) * 2);
    }

    if (!use_zlib) {
        if (!Write(tsdf, fid, "TSDF")) {
            return false;
        }
        if (!Write(weight, fid, "weight")) {
            return false;
        }
        if (!Write(color, fid, "color")) {
            return false;
        }
    } else {
        if (!CompressAndWrite(compressed_buf, tsdf, fid, "TSDF")) {
            return false;
        }
        if (!CompressAndWrite(compressed_buf, weight, fid, "weight")) {
            return false;
        }
        if (!CompressAndWrite(compressed_buf, color, fid, "color")) {
            return false;
        }
    }

    fclose(fid);
    return true;
}

bool ReadUniformTSDFVolumeFromBIN(const std::string &filename,
                                  cuda::UniformTSDFVolumeCuda &volume,
                                  bool use_zlib) {
    utility::LogInfo("Reading volume...\n");

    FILE *fid = fopen(filename.c_str(), "rb");
    if (fid == NULL) {
        utility::LogWarning("Read BIN failed: unable to open file: %s\n",
                              filename.c_str());
        return false;
    }

    /** metadata **/
    int volume_resolution;
    if (fread(&volume_resolution, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read num volumes\n");
        return false;
    }
    assert(volume_resolution == volume.N_);
    int N = volume.N_;
    int NNN = N * N * N;

    if (fread(&volume.voxel_length_, sizeof(float), 1, fid) < 1) {
        utility::LogWarning(
                "Read BIN failed: unable to read voxel size\n");
        return false;
    }
    Eigen::Matrix4d volume_to_world;
    float vij;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fread(&vij, sizeof(float), 1, fid) < 1) {
                utility::LogWarning(
                        "Read BIN failed: unable to read transform[%d, %d]\n",
                        i, j);
                return false;
            }
            volume_to_world(i, j) = vij;
        }
    }
    volume.transform_volume_to_world_.FromEigen(volume_to_world);

    /** values **/
    std::vector<float> tsdf_buf(NNN);
    std::vector<uchar> weight_buf(NNN);
    std::vector<cuda::Vector3b> color_buf(NNN);
    std::vector<uchar> compressed_buf(NNN * sizeof(float) * 2);

    if (!use_zlib) {
        if (!Read(tsdf_buf, fid, "TSDF")) {
            return false;
        }
        if (!Read(weight_buf, fid, "weight")) {
            return false;
        }
        if (!Read(color_buf, fid, "color")) {
            return false;
        }
    } else {
        if (!ReadAndUncompress(compressed_buf, tsdf_buf, fid, "TSDF")) {
            return false;
        }
        if (!ReadAndUncompress(compressed_buf, weight_buf, fid, "weight")) {
            return false;
        }
        if (!ReadAndUncompress(compressed_buf, color_buf, fid, "uchar")) {
            return false;
        }
    }

    volume.UploadVolume(tsdf_buf, weight_buf, color_buf);

    fclose(fid);
    return true;
}
}  // namespace io
}  // namespace open3d