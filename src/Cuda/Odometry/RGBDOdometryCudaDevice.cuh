//
// Created by wei on 10/1/18.
//

#pragma once

#include "RGBDOdometryCuda.h"

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Geometry/ImageCudaDevice.cuh>
#include <Cuda/Container/ArrayCudaDevice.cuh>

namespace open3d {
namespace cuda {
/**
 * Server end
 */
template<size_t N>
__device__
bool RGBDOdometryCudaDevice<N>::ComputePixelwiseCorrespondence(
    int x_source, int y_source, size_t level,
    int &x_target, int &y_target,
    Vector3f &X_source_on_target, float &d_target) {

    bool mask = false;
    Vector2f p_warpedf;
    Vector2i p_warped;

    if(odometry_type_ == OdometryType::FRAME_TO_MODEL) {

        /** Check 1: Is the source vertex and normal valid */
        mask = !source_vertex_[level].at(x_source, y_source).IsNaN();
        mask = mask && !source_normal_[level].at(x_source, y_source).IsNaN();
        /* printf("%d, [%f, %f, %f] [%f, %f, %f]\n", mask, curr_vertex(0), curr_vertex(1), curr_vertex(2), */
        /*         curr_normal(0), curr_normal(1), curr_normal(2)); */
        if (!mask) return false;

        /** Check 2: Is the projected vertex valid in image */
        X_source_on_target = transform_source_to_target_ *
                             source_vertex_[level].at(x_source, y_source);
        p_warpedf = intrinsics_[level].ProjectPoint(X_source_on_target);
        p_warped = Vector2i(int(p_warpedf(0) + 0.5f), int(p_warpedf(1) + 0.5f));
        /* printf("X_source_on_target: [%f, %f, %f], [%d, %d]\n", X_source_on_target(0), X_source_on_target(1), */
        /*         X_source_on_target(2), p_warped(0), p_warped(1)); */

        mask = intrinsics_[level].IsPixelValid(p_warped);
        if (!mask) return false;

        /** Check 3: Whether transformed point has reasonable z value */
        mask = IsValidDepth(X_source_on_target(2));
        if(!mask) return false;
        //! We don't use d_target
        d_target = 0.0f;
    }
    else {
        /** Check 1: depth valid in source? **/
        float d_source = source_depth_[level].at(x_source, y_source)(0);

        mask = IsValidDepth(d_source);
        if (!mask) return false;

        /** Check 2: reprojected point in image? **/
        X_source_on_target = transform_source_to_target_
            * intrinsics_[level].InverseProjectPixel(
                Vector2i(x_source, y_source), d_source);

        p_warpedf = intrinsics_[level].ProjectPoint(X_source_on_target);
        p_warped = Vector2i(int(p_warpedf(0) + 0.5f), int(p_warpedf(1) + 0.5f));

        mask = intrinsics_[level].IsPixelValid(p_warped);
        if (!mask) return false;

        /** Check 3: depth valid in target? Occlusion? -> 1ms **/
        d_target = target_depth_[level].at(p_warped(0), p_warped(1))(0);
        mask = IsValidDepth(d_target)
            && IsValidDepthDiff(d_target - X_source_on_target(2));
        if (!mask) return false;
    }

    x_target = p_warped(0);
    y_target = p_warped(1);

    return true;
}

template<size_t N>
__device__
bool RGBDOdometryCudaDevice<N>::ComputePixelwiseJacobianAndResidual(
    int x_source, int y_source, int x_target, int y_target, size_t level,
    const Vector3f &X_source_on_target, const float &d_target,
    Vector6f &jacobian_I, Vector6f &jacobian_D,
    float &residual_I, float &residual_D) {

    /********** Phase 2: Build linear system **********/
    /** Checks passed, let's rock! -> 3ms, can be 2ms faster if we don't use
     * interpolation
     *  \partial D(p_warped) \partial p_warped: [dx_D, dy_D] at p_warped, 1x2
     *  \partial I(p_warped) \partial p_warped: [dx_I, dy_I] at p_warped, 1x2
     *  \partial X.z \partial X: [0, 0, 1], 1x3
     *  \partial p_warped \partial X: [fx/Z, 0, -fx X/Z^2;
     *                                 0, fy/Z, -fy Y/Z^2]            2x3
     *  \partial X \partial \xi: [I | -[X]^] = [1 0 0 0  Z -Y;
     *                                          0 1 0 -Z 0 X;
     *                                          0 0 1 Y -X 0]         3x6
     * J_I = (d I(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     * J_D = (d D(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
     *     - (d X.z / d X) (d X / d \xi)
     */
    //! Intensity computations
    const float kSobelFactor = 0.125f;
    float dx_I = kSobelFactor * target_intensity_dx_[level].at(
        x_target, y_target)(0);
    float dy_I = kSobelFactor * target_intensity_dy_[level].at(
        x_target, y_target)(0);

    float fx = intrinsics_[level].fx_;
    float fy = intrinsics_[level].fy_;
    float inv_Z = 1.0f / X_source_on_target(2);
    float fx_on_Z = fx * inv_Z;
    float fy_on_Z = fy * inv_Z;

    float c0 = dx_I * fx_on_Z;
    float c1 = dy_I * fy_on_Z;
    float c2 = -(c0 * X_source_on_target(0) + c1 * X_source_on_target(1)) * inv_Z;

    jacobian_I(0) = sqrt_coeff_I_ * (-X_source_on_target(2) * c1 + X_source_on_target(1) * c2);
    jacobian_I(1) = sqrt_coeff_I_ * ( X_source_on_target(2) * c0 - X_source_on_target(0) * c2);
    jacobian_I(2) = sqrt_coeff_I_ * (-X_source_on_target(1) * c0 + X_source_on_target(0) * c1);
    jacobian_I(3) = sqrt_coeff_I_ * c0;
    jacobian_I(4) = sqrt_coeff_I_ * c1;
    jacobian_I(5) = sqrt_coeff_I_ * c2;

    residual_I = sqrt_coeff_I_ * (
        target_intensity_[level].at(x_target, y_target)(0) -
            source_intensity_[level].at(x_source, y_source)(0));

    /* printf("- (%d, %d), (%d, %d) -> " */
    /*        "(%f %f %f %f %f %f) - %f\n", */
    /*        x_source, y_source, x_target, y_target, */
    /*       jacobian_I(0), jacobian_I(1), jacobian_I(2), */
    /*       jacobian_I(3), jacobian_I(4), jacobian_I(5), residual_I); */

    //! Depth computations
    if(odometry_type_  == OdometryType::FRAME_TO_FRAME) {

        float dx_D = kSobelFactor * target_depth_dx_[level].at(
            x_target, y_target)(0);
        float dy_D = kSobelFactor * target_depth_dy_[level].at(
            x_target, y_target)(0);

        if (isnan(dx_D)) dx_D = 0;
        if (isnan(dy_D)) dy_D = 0;

        float d0 = dx_D * fx_on_Z;
        float d1 = dy_D * fy_on_Z;
        float d2 = -(d0 * X_source_on_target(0) + d1 * X_source_on_target(1)) * inv_Z;

        jacobian_D(0) = sqrt_coeff_D_ *
            ((-X_source_on_target(2) * d1 + X_source_on_target(1) * d2) - X_source_on_target(1));
        jacobian_D(1) = sqrt_coeff_D_ *
            ((X_source_on_target(2) * d0 - X_source_on_target(0) * d2) + X_source_on_target(0));
        jacobian_D(2) = sqrt_coeff_D_ *
            (-X_source_on_target(1) * d0 + X_source_on_target(0) * d1);
        jacobian_D(3) = sqrt_coeff_D_ * d0;
        jacobian_D(4) = sqrt_coeff_D_ * d1;
        jacobian_D(5) = sqrt_coeff_D_ * (d2 - 1.0f);

        residual_D = sqrt_coeff_D_ * (d_target - X_source_on_target(2));

    }
    else {
        Vector3f normal_t = target_normal_[level].at(x_target, y_target);
        Vector3f vertex_t = target_vertex_[level].at(x_target, y_target);

        if(vertex_t.IsNaN() || normal_t.IsNaN())
            return false;

        bool mask = IsValidDepth(vertex_t(2));
        if(!mask) return false;

        jacobian_D(0) = sqrt_coeff_D_ * (-X_source_on_target(2) * normal_t(1) + X_source_on_target(1) * normal_t(2));
        jacobian_D(1) = sqrt_coeff_D_ * ( X_source_on_target(2) * normal_t(0) - X_source_on_target(0) * normal_t(2));
        jacobian_D(2) = sqrt_coeff_D_ * (-X_source_on_target(1) * normal_t(0) + X_source_on_target(0) * normal_t(1));
        jacobian_D(3) = sqrt_coeff_D_ * normal_t(0);
        jacobian_D(4) = sqrt_coeff_D_ * normal_t(1);
        jacobian_D(5) = sqrt_coeff_D_ * normal_t(2);

        residual_D = sqrt_coeff_D_ * (X_source_on_target - vertex_t).dot(normal_t);
    }
    return true;
}


template<size_t N>
__device__
bool RGBDOdometryCudaDevice<N>::
    ComputePixelwiseCorrespondenceAndInformationJacobian(
    int x_source, int y_source,
    Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z) {

    bool mask = false;
    Vector3f X_target;
    if(odometry_type_ == OdometryType::FRAME_TO_MODEL) {
        mask = IsValidDepth(source_vertex_[0].at(x_source, y_source)(2));
        if (!mask) return false;

        Vector3f X_source_on_target = transform_source_to_target_
            * source_vertex_[0].at(x_source, y_source);

        Vector2f p_warpedf = intrinsics_[0].ProjectPoint(X_source_on_target);
        Vector2i p_warped(int(p_warpedf(0) + 0.5f), int(p_warpedf(1) + 0.5f));

        mask = intrinsics_[0].IsPixelValid(p_warpedf);
        if (!mask) return false;

        X_target = target_vertex_[0].at(p_warped(0), p_warped(1));
    }
    else {
        /** Check 1: depth valid in source? **/
        float d_source = source_depth_[0].at(x_source, y_source)(0);
        bool mask = IsValidDepth(d_source);
        if (!mask) return false;

        /** Check 2: reprojected point in image? **/
        Vector3f X_source_on_target = transform_source_to_target_
            * intrinsics_[0].InverseProjectPixel(
                Vector2i(x_source, y_source), d_source);

        Vector2f p_warpedf = intrinsics_[0].ProjectPoint(X_source_on_target);
        mask = intrinsics_[0].IsPixelValid(p_warpedf);
        if (!mask) return false;

        Vector2i p_warped(int(p_warpedf(0) + 0.5f), int(p_warpedf(1) + 0.5f));

        /** Check 3: depth valid in target? Occlusion? -> 1ms **/
        float d_target = target_depth_[0].at(p_warped(0), p_warped(1))(0);
        mask = IsValidDepth(d_target) && IsValidDepthDiff(
            d_target - X_source_on_target(2));
        if (!mask) return false;

        X_target = intrinsics_[0].InverseProjectPixel(p_warped, d_target);
    }

    jacobian_x(0) = jacobian_x(4) = jacobian_x(5) = 0;
    jacobian_x(1) = X_target(2);
    jacobian_x(2) = -X_target(1);
    jacobian_x(3) = 1;

    jacobian_y(1) = jacobian_y(3) = jacobian_y(5) = 0;
    jacobian_y(0) = -X_target(2);
    jacobian_y(2) = X_target(0);
    jacobian_y(4) = 1.0f;

    jacobian_z(2) = jacobian_z(3) = jacobian_z(4) = 0;
    jacobian_z(0) = X_target(1);
    jacobian_z(1) = -X_target(0);
    jacobian_z(5) = 1.0f;

    return true;
}
} // cuda
} // open3d
