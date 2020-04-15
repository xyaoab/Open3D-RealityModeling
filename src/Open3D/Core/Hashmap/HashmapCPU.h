#include "Hashmap.h"

namespace open3d {
class CPUHashmap : public Hashmap {
public:
    ~CPUHashmap();

    CPUHashmap(uint32_t max_keys,
               uint32_t dsize_key,
               uint32_t dsize_value,
               open3d::Device device,
               hash_t hash_fn_ptr);

    std::pair<iterator_t*, uint8_t*> Insert(uint8_t* input_keys,
                                            uint8_t* input_values,
                                            uint32_t input_key_size);

    std::pair<iterator_t*, uint8_t*> Search(uint8_t* input_keys,
                                            uint32_t input_key_size);

    uint8_t* Remove(uint8_t* input_keys, uint32_t input_key_size);
};
}  // namespace open3d
