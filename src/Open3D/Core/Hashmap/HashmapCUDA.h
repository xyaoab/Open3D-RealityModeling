/*
 * Copyright 2019 Saman Ashkiani
 * Rewrite 2019 by Wei Dong
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <memory>

#include <thrust/pair.h>

#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/MemoryManager.h"

#include "InternalMemoryManager.h"
#include "InternalNodeManager.h"

/// Internal Hashtable Node: (31 units and 1 next ptr) representation.
/// \member kv_pair_ptrs:
/// Each element is an internal ptr to a kv pair managed by the
/// InternalMemoryManager. Can be converted to a real ptr.
/// \member next_slab_ptr:
/// An internal ptr managed by InternalNodeManager.

#define MAX_KEY_BYTESIZE 32

template <typename Key, typename Value>
struct Pair {
    Key first;
    Value second;
    __device__ __host__ Pair() {}
    __device__ __host__ Pair(const Key& key, const Value& value)
        : first(key), second(value) {}
};

template <typename Key, typename Value>
__device__ __host__ Pair<Key, Value> make_pair(const Key& key,
                                               const Value& value) {
    return Pair<Key, Value>(key, value);
}

class Slab {
public:
    ptr_t kv_pair_ptrs[31];
    ptr_t next_slab_ptr;
};

typedef uint64_t (*hash_t)(uint8_t*, uint32_t);

class HashmapCUDAContext {
public:
    HashmapCUDAContext();
    __host__ void Setup(Slab* bucket_list_head,
                        const uint32_t num_buckets,
                        const uint32_t dsize_key,
                        const uint32_t dsize_value,
                        const InternalNodeManagerContext& allocator_ctx,
                        const InternalMemoryManagerContext& pair_allocator_ctx);

    /* Core SIMT operations, shared by both simplistic and verbose
     * interfaces */
    __device__ Pair<ptr_t, uint8_t> Insert(uint8_t& lane_active,
                                           const uint32_t lane_id,
                                           const uint32_t bucket_id,
                                           uint8_t* key_ptr,
                                           uint8_t* value_ptr);

    __device__ Pair<ptr_t, uint8_t> Search(uint8_t& lane_active,
                                           const uint32_t lane_id,
                                           const uint32_t bucket_id,
                                           uint8_t* key_ptr);

    __device__ uint8_t Remove(uint8_t& lane_active,
                              const uint32_t lane_id,
                              const uint32_t bucket_id,
                              uint8_t* key_ptr);

    /* Hash function */
    __device__ __host__ uint32_t ComputeBucket(uint8_t* key_ptr) const;
    __device__ __host__ uint32_t bucket_size() const { return num_buckets_; }

    __device__ __host__ InternalNodeManagerContext& get_slab_alloc_ctx() {
        return slab_list_allocator_ctx_;
    }
    __device__ __host__ InternalMemoryManagerContext& get_pair_alloc_ctx() {
        return pair_allocator_ctx_;
    }

    __device__ __forceinline__ ptr_t* get_unit_ptr_from_list_nodes(
            const ptr_t slab_ptr, const uint32_t lane_id) {
        return slab_list_allocator_ctx_.get_unit_ptr_from_slab(slab_ptr,
                                                               lane_id);
    }
    __device__ __forceinline__ ptr_t* get_unit_ptr_from_list_head(
            const uint32_t bucket_id, const uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(bucket_list_head_) +
               bucket_id * BASE_UNIT_SIZE + lane_id;
    }

private:
    __device__ __forceinline__ void WarpSyncKey(uint8_t* key_ptr,
                                                const uint32_t lane_id,
                                                uint8_t* ret_key_ptr);
    __device__ __forceinline__ int32_t WarpFindKey(uint8_t* src_key_ptr,
                                                   const uint32_t lane_id,
                                                   const uint32_t ptr);
    __device__ __forceinline__ int32_t WarpFindEmpty(const uint32_t unit_data);

    __device__ __forceinline__ ptr_t AllocateSlab(const uint32_t lane_id);
    __device__ __forceinline__ void FreeSlab(const ptr_t slab_ptr);

public:
    uint32_t num_buckets_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

    hash_t hash_fn_;

    Slab* bucket_list_head_;
    InternalNodeManagerContext slab_list_allocator_ctx_;
    InternalMemoryManagerContext pair_allocator_ctx_;
};

class HashmapCUDA {
public:
    using MemMgr = open3d::MemoryManager;
    HashmapCUDA(const uint32_t max_bucket_count,
                const uint32_t max_keyvalue_count,
                const uint32_t dsize_key,
                const uint32_t dsize_value,
                open3d::Device device);

    ~HashmapCUDA();

    void Insert(uint8_t* input_keys,
                uint8_t* input_values,
                iterator_t* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Search(uint8_t* input_keys,
                iterator_t* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Remove(uint8_t* input_keys, uint8_t* output_masks, uint32_t num_keys);

    /// Parallel collect all the iterators from begin to end
    void GetIterators(iterator_t* iterators, uint32_t& num_iterators);

    /// Parallel extract keys and values from iterators
    void ExtractIterators(iterator_t* iterators,
                          uint8_t* keys,
                          uint8_t* values,
                          uint32_t num_iterators);

    /// Profiler
    std::vector<int> CountElemsPerBucket();
    double ComputeLoadFactor();

private:
    Slab* bucket_list_head_;
    uint32_t num_buckets_;

    HashmapCUDAContext gpu_context_;

    std::shared_ptr<InternalMemoryManager<MemMgr>> pair_allocator_;
    std::shared_ptr<InternalNodeManager<MemMgr>> slab_list_allocator_;

    open3d::Device device_;
};
