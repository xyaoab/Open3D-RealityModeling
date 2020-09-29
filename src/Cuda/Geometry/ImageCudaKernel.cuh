#include "ImageCudaDevice.cuh"
#include <math_constants.h>

namespace open3d {

namespace cuda {
/**
 * Downsampling
 */
template<typename Scalar, size_t Channel>
__global__
void DownsampleKernel(ImageCudaDevice<Scalar, Channel> src,
                      ImageCudaDevice<Scalar, Channel> dst,
                      DownsampleMethod method) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    /** Re-write the function by hard-coding if we want to accelerate **/
    switch (method) {
        case BoxFilter:
            dst.at(u, v) = src.BoxFilter2x2(u << 1, v << 1);
            return;
        case BoxFilterNormalize:
            dst.at(u, v) = src.BoxFilter2x2(u << 1, v << 1, true);
            return;
        case GaussianFilter:
            dst.at(u, v) = src.GaussianFilter(u << 1, v << 1, Gaussian3x3);
            return;
        default:printf("Unsupported method.\n");
    }
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::Downsample(
    ImageCuda<Scalar, Channel> &src, ImageCuda<Scalar, Channel> &dst,
    DownsampleMethod method) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(dst.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    DownsampleKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, method);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Obtain vertex map
 */
template<typename Scalar, size_t Channel>
__global__
void GetVertexMapKernel(ImageCudaDevice<Scalar, Channel> src,
                        ImageCudaDevice<float, 3> dst,
                        PinholeCameraIntrinsicCuda intrinsic) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.width_ || y >= dst.height_)
        return;

    float depth = src.at(x, y, 0);
    if(depth == 0 || isnan(depth))
    {
        dst.at(x, y) = Vector3f(nanf("nan"));
        return;
    }

    Vector3f point = intrinsic.InverseProjectPixel(Vector2i(x, y), depth);
    dst.at(x, y) = point;
}


template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::GetVertexMap(
    ImageCuda<Scalar, Channel> &src, ImageCuda<float, 3> &dst,
    PinholeCameraIntrinsicCuda &intrinsic) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(dst.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GetVertexMapKernel<<<blocks, threads>>>(
            *src.device_, *dst.device_, intrinsic);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Scalar, size_t Channel>
__global__
void GetNormalMapKernel(ImageCudaDevice<Scalar, Channel> src,
                        ImageCudaDevice<float, 3> dst_normal_map) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= src.width_ || v >= src.height_)
        return;

    if (u == src.width_ - 1 || v == src.height_ - 1)
    {
        dst_normal_map.at(u, v) = Vector3f(nanf("nan"));
        return;
    }

    Vector3f v00, v01, v10;

    v00(0) = src.at(u, v, 0);
    v01(0) = src.at(u, v+1, 0);
    v10(0) = src.at(u+1, v, 0);

    v00(1) = src.at(u, v, 1);
    v01(1) = src.at(u, v+1, 1);
    v10(1) = src.at(u+1, v, 1);

    v00(2) = src.at(u, v, 2);
    v01(2) = src.at(u, v+1, 2);
    v10(2) = src.at(u+1, v, 2);

    if(v00.IsNaN() || v01.IsNaN() || v10.IsNaN())
    {
        dst_normal_map.at(u, v) = Vector3f(nanf("nan"));
        return;
    }

    Vector3f vec1 = v01 - v00;
    Vector3f vec2 = v10 - v00;
    Vector3f normal = (vec1.cross(vec2)).normalized();

    dst_normal_map.at(u, v) = normal;
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::GetNormalMap(
        ImageCuda<Scalar, Channel> &src, ImageCuda<float, 3> &dst_normal_map) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(dst_normal_map.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GetNormalMapKernel<<<blocks, threads>>>(*src.device_, *dst_normal_map.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Shift
 */
template<typename Scalar, size_t Channel>
__global__
void ShiftKernel(ImageCudaDevice<Scalar, Channel> src,
                 ImageCudaDevice<Scalar, Channel> dst,
                 float dx, float dy) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    dst.at(u, v) = VectorCuda<Scalar, Channel>(0);
    if (u >= dst.width_ || v >= dst.height_)
        return;

    if (u + dx < 0 || u + dx >= src.width_ - 1
        || v + dy < 0 || v + dy >= src.height_ - 1)
        return;

    dst.at(u, v) = src.interp_at(u + dx, v + dy);
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::Shift(
    ImageCuda<Scalar, Channel> &src, ImageCuda<Scalar, Channel> &dst,
    float dx, float dy) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ShiftKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, dx, dy);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gaussian
 */
template<typename Scalar, size_t Channel>
__global__
void GaussianKernel(ImageCudaDevice<Scalar, Channel> src,
                    ImageCudaDevice<Scalar, Channel> dst,
                    int kernel_idx) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    dst.at(u, v) = src.GaussianFilter(u, v, kernel_idx);
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::Gaussian(
    ImageCuda<Scalar, Channel> &src, ImageCuda<Scalar, Channel> &dst,
    int kernel_idx) {

    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    GaussianKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, kernel_idx);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Bilateral
 */
template<typename Scalar, size_t Channel>
__global__
void BilateralKernel(ImageCudaDevice<Scalar, Channel> src,
                     ImageCudaDevice<Scalar, Channel> dst,
                     int kernel_idx,
                     float val_sigma) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= dst.width_ || v >= dst.height_)
        return;

    dst.at(u, v) = src.BilateralFilter(u, v, kernel_idx, val_sigma);
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::Bilateral(
    ImageCuda<Scalar, Channel> &src, ImageCuda<Scalar, Channel> &dst,
    int kernel_idx, float val_sigma) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BilateralKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, kernel_idx, val_sigma);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Gradient, using Sobel operator
 */
template<typename Scalar, size_t Channel>
__global__
void SobelKernel(ImageCudaDevice<Scalar, Channel> src,
                 ImageCudaDevice<float, Channel> dx,
                 ImageCudaDevice<float, Channel> dy) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= src.width_ - 1|| v >= src.height_ - 1) return;
    if (u <= 0 || v <= 0) return;

    typename ImageCudaDevice<Scalar, Channel>::Grad grad;

    grad = src.Sobel(u, v);

    dx(u, v) = grad.dx;
    dy(u, v) = grad.dy;
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::Sobel(
    ImageCuda<Scalar, Channel> &src,
    ImageCuda<float, Channel> &dx, ImageCuda<float, Channel> &dy) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    SobelKernel << < blocks, threads >> > (
        *src.device_, *dx.device_, *dy.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/**
 * Conversion
 */
template<typename Scalar, size_t Channel>
__global__
void ConvertToFloatKernel(ImageCudaDevice<Scalar, Channel> src,
                          ImageCudaDevice<float, Channel> dst,
                          float scale, float offset) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dst.width_ || v >= dst.height_) return;

    dst.at(u, v) = src.at(u, v).template cast<float>() * scale
        + VectorCuda<float, Channel>(offset);
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::ConvertToFloat(
    ImageCuda<Scalar, Channel> &src,
    ImageCuda<float, Channel> &dst,
    float scale, float offset) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ConvertToFloatKernel << < blocks, threads >> > (
        *src.device_, *dst.device_, scale, offset);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<typename Scalar, size_t Channel>
__global__
void ConvertRGBToIntensityKernel(ImageCudaDevice<Scalar, Channel> src,
                                 ImageCudaDevice<float, 1> dst) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= dst.width_ || v >= dst.height_) return;

    VectorCuda<Scalar, Channel> &rgb = src.at(u, v);
    dst.at(u, v) = Vector1f(
        (0.2990f * rgb(0) + 0.5870f * rgb(1) + 0.1140f * rgb(2)) / 255.0f);
}

template<typename Scalar, size_t Channel>
__host__
void ImageCudaKernelCaller<Scalar, Channel>::ConvertRGBToIntensity(
    ImageCuda<Scalar, Channel> &src, ImageCuda<float, 1> &dst) {
    const dim3 blocks(DIV_CEILING(src.width_, THREAD_2D_UNIT),
                      DIV_CEILING(src.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ConvertRGBToIntensityKernel << < blocks, threads >> > (
        *src.device_, *dst.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d
