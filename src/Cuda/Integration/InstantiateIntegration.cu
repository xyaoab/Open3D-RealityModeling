//
// Created by wei on 10/10/18.
//

#include <Cuda/Container/HashTableCudaDevice.cuh>
#include <Cuda/Container/HashTableCudaKernel.cuh>

#include <Cuda/Integration/UniformTSDFVolumeCudaDevice.cuh>
#include <Cuda/Integration/UniformTSDFVolumeCudaKernel.cuh>
#include <Cuda/Integration/UniformMeshVolumeCudaDevice.cuh>
#include <Cuda/Integration/UniformMeshVolumeCudaKernel.cuh>
#include <Cuda/Integration/ScalableTSDFVolumeCudaDevice.cuh>
#include <Cuda/Integration/ScalableTSDFVolumeCudaKernel.cuh>
#include <Cuda/Integration/ScalableMeshVolumeCudaDevice.cuh>
#include <Cuda/Integration/ScalableMeshVolumeCudaKernel.cuh>
#include <Cuda/Integration/RGBDToTSDFKernel.cuh>

#include <Cuda/Experiment/ScalableTSDFVolumeProcessorCudaKernel.cuh>
#include <Cuda/Experiment/ScalableVolumeRegistrationCudaDevice.cuh>
#include <Cuda/Experiment/ScalableVolumeRegistrationCudaKernel.cuh>

namespace open3d {
namespace cuda {
template
class HashTableCudaKernelCaller
    <Vector3i, UniformTSDFVolumeCudaDevice, SpatialHasher>;
template
class MemoryHeapCudaKernelCaller<UniformTSDFVolumeCudaDevice>;

}
}