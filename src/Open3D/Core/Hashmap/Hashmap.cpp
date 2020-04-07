#include "Hashmap.h"
#include <unordered_map>

namespace cuda {

std::shared_ptr<CUDAHashmap> CreateCUDAHashmap(uint32_t max_keys,
                                               uint32_t dsize_key,
                                               uint32_t dsize_value,
                                               open3d::Device device) {
    return std::make_shared<CUDAHashmap>(max_keys, dsize_key, dsize_value,
                                         device);
}

std::shared_ptr<Hashmap> CreateHashmap(uint32_t max_keys,
                                       uint32_t dsize_key,
                                       uint32_t dsize_value,
                                       open3d::Device device) {
    static std::unordered_map<
            open3d::Device::DeviceType,
            std::function<std::shared_ptr<Hashmap>(uint32_t, uint32_t, uint32_t,
                                                   open3d::Device)>,
            open3d::utility::hash_enum_class::hash>
            map_device_type_to_memory_manager = {
                    {open3d::Device::DeviceType::CUDA, CreateCUDAHashmap}};

    if (map_device_type_to_memory_manager.find(device.GetType()) ==
        map_device_type_to_memory_manager.end()) {
        open3d::utility::LogError(
                "MemoryManager::GetDeviceMemoryManager: Unimplemented device");
    }

    auto fn = map_device_type_to_memory_manager.at(device.GetType());
    return fn(max_keys, dsize_key, dsize_value, device);
}
}  // namespace cuda
