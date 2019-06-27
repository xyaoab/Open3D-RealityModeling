//
// Created by dongw1 on 6/25/19.
//

#pragma once

#include <zlib.h>

namespace open3d {
namespace io {

template <typename T>
bool CompressAndWrite(std::vector<uchar> &compressed_buf,
                      const std::vector<T> &src,
                      FILE *&fid,
                      const std::string &msg) {
    size_t compressed_len = compressed_buf.size();
    if (Z_OK != compress2(compressed_buf.data(), &compressed_len,
                          (uchar *)src.data(), src.size() * sizeof(T),
                          Z_BEST_SPEED)) {
        utility::PrintWarning("Compressing %s failed\n", msg.c_str());
    }
    if (fwrite(&compressed_len, sizeof(size_t), 1, fid) < 1) {
        utility::PrintWarning("Write BIN failed: unable to write %s bytes\n",
                              msg.c_str());
        return false;
    }
    if (fwrite(compressed_buf.data(), sizeof(uchar), compressed_len, fid) <
        compressed_len) {
        utility::PrintWarning("Write BIN failed: unable to write %s data\n",
                              msg.c_str());
        return false;
    }

    return true;
}

template <typename T>
bool Write(const std::vector<T> &src, FILE *&fid, const std::string &msg) {
    if (fwrite(src.data(), sizeof(T), src.size(), fid) < src.size()) {
        utility::PrintWarning("Write BIN failed: unable to write %s\n",
                              msg.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool Read(std::vector<T> &dst, FILE *&fid, const std::string &msg) {
    if (fread(dst.data(), sizeof(T), dst.size(), fid) < dst.size()) {
        utility::PrintWarning("Read BIN failed: unable to read %s\n",
                              msg.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool ReadAndUncompress(std::vector<uchar> &compressed_buf,
                       std::vector<T> &dst,
                       FILE *&fid,
                       const std::string &msg) {
    size_t compressed_len;
    if (fread(&compressed_len, sizeof(size_t), 1, fid) < 1) {
        utility::PrintWarning("Read BIN failed: unable to read %s bytes\n",
                              msg.c_str());
        return false;
    }
    if (fread(compressed_buf.data(), sizeof(uchar), compressed_len, fid) <
        compressed_len) {
        utility::PrintWarning("Read BIN failed: unable to read %s data\n",
                              msg.c_str());
        return false;
    }

    size_t uncompressed_len = dst.size() * sizeof(float);
    if (Z_OK != uncompress((uchar *)dst.data(), &uncompressed_len,
                           compressed_buf.data(), compressed_len)) {
        utility::PrintWarning("Uncompressing %s failed\n", msg.c_str());
        return false;
    }

    return true;
}
}  // namespace io
}  // namespace open3d
