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

#include <thrust/pair.h>
#include <cassert>
#include <memory>
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/MemoryManager.h"
#include "memory_alloc.h"
#include "slab_alloc.h"

/**
 * Interface
 **/
class Slab {
public:
    ptr_t pair_ptrs[31];
    ptr_t next_slab_ptr;
};

template <typename Key, typename Value>
struct _Pair {
    Key first;
    Value second;
    __device__ __host__ _Pair(const Key& key, const Value& value)
        : first(key), second(value) {}
    __device__ __host__ _Pair() : first(), second() {}
};

template <typename Key, typename Value>
__device__ __host__ _Pair<Key, Value> _make_pair(const Key& key,
                                                 const Value& value) {
    return _Pair<Key, Value>(key, value);
}

template <typename Key, typename Value, typename Hash>
class SlabHashContext;

template <typename Key, typename Value>
using _Iterator = _Pair<Key, Value>*;

template <typename Key, typename Value, typename Hash, class Alloc>
class SlabHash {
public:
    SlabHash(const uint32_t max_bucket_count,
             const uint32_t max_keyvalue_count,
             open3d::Device device);

    ~SlabHash();

    /* Verbose output (similar to std): return success masks for all operations,

    * and iterators for insert and search (not for remove operation, as
     * iterators are invalid after erase).
     * Output iterators supports READ/WRITE: change to these output will
     * DIRECTLY change the internal hash table.
     */
    void Insert(Key* input_keys,
                Value* input_values,
                _Iterator<Key, Value>* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Search(Key* input_keys,
                _Iterator<Key, Value>* output_iterators,
                uint8_t* output_masks,
                uint32_t num_keys);
    void Remove(Key* input_keys, uint8_t* output_masks, uint32_t num_keys);

    /* Parallel collect all the iterators from begin to end */
    void GetIterators(_Iterator<Key, Value>* iterators,
                      uint32_t& num_iterators);
    /* Parallel extract keys and values from iterators */
    void ExtractIterators(_Iterator<Key, Value>* iterators,
                          Key* keys,
                          Value* values,
                          uint32_t num_iterators);

    /* Debug usages */
    std::vector<int> CountElemsPerBucket();

    double ComputeLoadFactor();

private:
    Slab* bucket_list_head_;
    uint32_t num_buckets_;

    SlabHashContext<Key, Value, Hash> gpu_context_;

    std::shared_ptr<MemoryAlloc<_Pair<Key, Value>, Alloc>> pair_allocator_;
    std::shared_ptr<SlabAlloc<Alloc>> slab_list_allocator_;
    open3d::Device device_;
};

/** Verbose version **/
template <typename Key, typename Value, typename Hash>
__global__ void InsertKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* input_keys,
                             Value* input_values,
                             _Iterator<Key, Value>* output_iterators,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename Key, typename Value, typename Hash>
__global__ void SearchKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* input_keys,
                             _Iterator<Key, Value>* output_iterators,
                             uint8_t* output_masks,
                             uint32_t num_keys);
template <typename Key, typename Value, typename Hash>
__global__ void RemoveKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* input_keys,
                             uint8_t* output_masks,
                             uint32_t num_keys);

template <typename Key, typename Value, typename Hash>
__global__ void GetIteratorsKernel(
        SlabHashContext<Key, Value, Hash> slab_hash_ctx,
        _Iterator<Key, Value>* output_iterators,
        uint32_t* output_iterator_count,
        uint32_t num_buckets);
template <typename Key, typename Value, typename Hash>
__global__ void CountElemsPerBucketKernel(
        SlabHashContext<Key, Value, Hash> slab_hash_ctx,
        uint32_t* bucket_elem_counts);

/**
 * Implementation for the host class
 **/
template <typename Key, typename Value, typename Hash, class Alloc>
SlabHash<Key, Value, Hash, Alloc>::SlabHash(const uint32_t max_bucket_count,
                                            const uint32_t max_keyvalue_count,
                                            open3d::Device device)
    : num_buckets_(max_bucket_count),
      device_(device),
      bucket_list_head_(nullptr) {
    // allocate an initialize the allocator:
    pair_allocator_ = std::make_shared<MemoryAlloc<_Pair<Key, Value>, Alloc>>(
            max_keyvalue_count, device_);
    slab_list_allocator_ = std::make_shared<SlabAlloc<Alloc>>(device_);

    // allocating initial buckets:
    bucket_list_head_ = static_cast<Slab*>(
            Alloc::Malloc(num_buckets_ * sizeof(Slab), device_));
    OPEN3D_CUDA_CHECK(
            cudaMemset(bucket_list_head_, 0xFF, sizeof(Slab) * num_buckets_));

    gpu_context_.Setup(bucket_list_head_, num_buckets_,
                       slab_list_allocator_->getContext(),
                       pair_allocator_->gpu_context_);
}

template <typename Key, typename Value, typename Hash, class Alloc>
SlabHash<Key, Value, Hash, Alloc>::~SlabHash() {
    Alloc::Free(bucket_list_head_, device_);
}

template <typename Key, typename Value, typename Hash, class Alloc>
void SlabHash<Key, Value, Hash, Alloc>::Insert(Key* keys,
                                               Value* values,
                                               _Iterator<Key, Value>* iterators,
                                               uint8_t* masks,
                                               uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    InsertKernel<Key, Value, Hash><<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, values, iterators, masks, num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Value, typename Hash, class Alloc>
void SlabHash<Key, Value, Hash, Alloc>::Search(Key* keys,
                                               _Iterator<Key, Value>* iterators,
                                               uint8_t* masks,
                                               uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    SearchKernel<Key, Value, Hash><<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, iterators, masks, num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Value, typename Hash, class Alloc>
void SlabHash<Key, Value, Hash, Alloc>::Remove(Key* keys,
                                               uint8_t* masks,
                                               uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    RemoveKernel<Key, Value, Hash>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, masks, num_keys);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

/* Debug usage */
template <typename Key, typename Value, typename Hash, class Alloc>
std::vector<int> SlabHash<Key, Value, Hash, Alloc>::CountElemsPerBucket() {
    auto elems_per_bucket_buffer = static_cast<uint32_t*>(
            Alloc::Malloc(num_buckets_ * sizeof(uint32_t), device_));

    thrust::device_vector<uint32_t> elems_per_bucket(
            elems_per_bucket_buffer, elems_per_bucket_buffer + num_buckets_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    CountElemsPerBucketKernel<Key, Value, Hash><<<num_blocks, blocksize>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()));

    std::vector<int> result(num_buckets_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    Alloc::Free(elems_per_bucket_buffer, device_);
    return std::move(result);
}

template <typename Key, typename Value, typename Hash, class Alloc>
double SlabHash<Key, Value, Hash, Alloc>::ComputeLoadFactor() {
    auto elems_per_bucket = CountElemsPerBucket();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    slab_list_allocator_->getContext() = gpu_context_.get_slab_alloc_ctx();
    auto slabs_per_bucket = slab_list_allocator_->CountSlabsPerSuperblock();
    int total_slabs_stored = std::accumulate(
            slabs_per_bucket.begin(), slabs_per_bucket.end(), num_buckets_);

    double load_factor = double(total_elems_stored) /
                         double(total_slabs_stored * WARP_WIDTH);

    return load_factor;
}

/**
 * Internal implementation for the device proxy:
 * DO NOT ENTER!
 **/
template <typename Key, typename Value, typename Hash>
class SlabHashContext {
public:
    SlabHashContext();
    __host__ void Setup(
            Slab* bucket_list_head,
            const uint32_t num_buckets,
            const SlabAllocContext& allocator_ctx,
            const MemoryAllocContext<_Pair<Key, Value>>& pair_allocator_ctx);

    /* Core SIMT operations, shared by both simplistic and verbose
     * interfaces */
    __device__ _Pair<ptr_t, uint8_t> Insert(uint8_t& lane_active,
                                            const uint32_t lane_id,
                                            const uint32_t bucket_id,
                                            const Key& key,
                                            const Value& value);

    __device__ _Pair<ptr_t, uint8_t> Search(uint8_t& lane_active,
                                            const uint32_t lane_id,
                                            const uint32_t bucket_id,
                                            const Key& key);

    __device__ uint8_t Remove(uint8_t& lane_active,
                              const uint32_t lane_id,
                              const uint32_t bucket_id,
                              const Key& key);

    /* Hash function */
    __device__ __host__ uint32_t ComputeBucket(const Key& key) const;
    __device__ __host__ uint32_t bucket_size() const { return num_buckets_; }

    __device__ __host__ SlabAllocContext& get_slab_alloc_ctx() {
        return slab_list_allocator_ctx_;
    }
    __device__ __host__ MemoryAllocContext<_Pair<Key, Value>>
    get_pair_alloc_ctx() {
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
    __device__ __forceinline__ void WarpSyncKey(const Key& key,
                                                const uint32_t lane_id,
                                                Key& ret);
    __device__ __forceinline__ int32_t WarpFindKey(const Key& src_key,
                                                   const uint32_t lane_id,
                                                   const uint32_t unit_data);
    __device__ __forceinline__ int32_t WarpFindEmpty(const uint32_t unit_data);

    __device__ __forceinline__ ptr_t AllocateSlab(const uint32_t lane_id);
    __device__ __forceinline__ void FreeSlab(const ptr_t slab_ptr);

private:
    uint32_t num_buckets_;
    Hash hash_fn_;

    Slab* bucket_list_head_;
    SlabAllocContext slab_list_allocator_ctx_;
    MemoryAllocContext<_Pair<Key, Value>> pair_allocator_ctx_;
};

/**
 * Definitions
 **/
template <typename Key, typename Value, typename Hash>
SlabHashContext<Key, Value, Hash>::SlabHashContext()
    : num_buckets_(0), bucket_list_head_(nullptr) {
    static_assert(sizeof(Slab) == (WARP_WIDTH * sizeof(ptr_t)),
                  "Invalid slab size");
}

template <typename Key, typename Value, typename Hash>
__host__ void SlabHashContext<Key, Value, Hash>::Setup(
        Slab* bucket_list_head,
        const uint32_t num_buckets,
        const SlabAllocContext& allocator_ctx,
        const MemoryAllocContext<_Pair<Key, Value>>& pair_allocator_ctx) {
    bucket_list_head_ = bucket_list_head;

    num_buckets_ = num_buckets;
    slab_list_allocator_ctx_ = allocator_ctx;
    pair_allocator_ctx_ = pair_allocator_ctx;
}

template <typename Key, typename Value, typename Hash>
__device__ __host__ __forceinline__ uint32_t
SlabHashContext<Key, Value, Hash>::ComputeBucket(const Key& key) const {
    return hash_fn_(key) % num_buckets_;
}

template <typename Key, typename Value, typename Hash>
__device__ __forceinline__ void SlabHashContext<Key, Value, Hash>::WarpSyncKey(
        const Key& key, const uint32_t lane_id, Key& ret) {
    const int chunks = sizeof(Key) / sizeof(int);
#pragma unroll 1
    for (size_t i = 0; i < chunks; ++i) {
        ((int*)(&ret))[i] = __shfl_sync(ACTIVE_LANES_MASK, ((int*)(&key))[i],
                                        lane_id, WARP_WIDTH);
    }
}

template <typename Key, typename Value, typename Hash>
__device__ int32_t SlabHashContext<Key, Value, Hash>::WarpFindKey(
        const Key& key, const uint32_t lane_id, const ptr_t ptr) {
    uint8_t is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & PAIR_PTR_LANES_MASK)
            /* validate key addrs */
            && (ptr != EMPTY_PAIR_PTR)
            /* find keys in memory heap */
            && pair_allocator_ctx_.extract(ptr).first == key;

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_found)) - 1;
}

template <typename Key, typename Value, typename Hash>
__device__ __forceinline__ int32_t
SlabHashContext<Key, Value, Hash>::WarpFindEmpty(const ptr_t ptr) {
    uint8_t is_lane_empty = (ptr == EMPTY_PAIR_PTR);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_empty)) - 1;
}

template <typename Key, typename Value, typename Hash>
__device__ __forceinline__ ptr_t
SlabHashContext<Key, Value, Hash>::AllocateSlab(const uint32_t lane_id) {
    return slab_list_allocator_ctx_.WarpAllocate(lane_id);
}

template <typename Key, typename Value, typename Hash>
__device__ __forceinline__ void SlabHashContext<Key, Value, Hash>::FreeSlab(
        const ptr_t slab_ptr) {
    slab_list_allocator_ctx_.FreeUntouched(slab_ptr);
}

template <typename Key, typename Value, typename Hash>
__device__ _Pair<ptr_t, uint8_t> SlabHashContext<Key, Value, Hash>::Search(
        uint8_t& to_search,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const Key& query_key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    ptr_t iterator = NULL_ITERATOR;
    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_search))) {
        /** 0. Restart from linked list head if the last query is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        Key src_key;
        WarpSyncKey(query_key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            ptr_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                to_search = false;

                iterator = found_pair_internal_ptr;
                mask = true;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab: lane 31 reads 'next' */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                if (lane_id == src_lane) {
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = next_slab_ptr;
            }
        }

        prev_work_queue = work_queue;
    }

    return _make_pair(iterator, mask);
}

/*
 * Insert: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYS
 */
template <typename Key, typename Value, typename Hash>
__device__ _Pair<ptr_t, uint8_t> SlabHashContext<Key, Value, Hash>::Insert(
        uint8_t& to_be_inserted,
        const uint32_t lane_id,
        const uint32_t bucket_id,
        const Key& key,
        const Value& value) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    ptr_t iterator = NULL_ITERATOR;
    uint8_t mask = false;

    /** WARNING: Allocation should be finished in warp,
     * results are unexpected otherwise **/
    int prealloc_pair_internal_ptr = EMPTY_PAIR_PTR;
    if (to_be_inserted) {
        prealloc_pair_internal_ptr = pair_allocator_ctx_.Allocate();
        pair_allocator_ctx_.extract(prealloc_pair_internal_ptr) =
                _make_pair(key, value);
    }

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);
        Key src_key;
        WarpSyncKey(key, src_lane, src_key);

        /* Each lane in the warp reads a uint in the slab */
        uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);
        int32_t lane_empty = WarpFindEmpty(unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
                pair_allocator_ctx_.Free(prealloc_pair_internal_ptr);
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                // TODO: check why we cannot put malloc here
                const uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_empty)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_empty);
                ptr_t old_pair_internal_ptr =
                        atomicCAS((unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR,
                                  prealloc_pair_internal_ptr);

                /** Branch 2.1: SUCCEED **/
                if (old_pair_internal_ptr == EMPTY_PAIR_PTR) {
                    to_be_inserted = false;

                    iterator = prealloc_pair_internal_ptr;
                    mask = true;
                }
                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_PTR) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_PTR_LANE) {
                    const uint32_t* unit_data_ptr =
                            (curr_slab_ptr == HEAD_SLAB_PTR)
                                    ? get_unit_ptr_from_list_head(
                                              src_bucket, NEXT_SLAB_PTR_LANE)
                                    : get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_PTR_LANE);

                    ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_PTR, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }

    return _make_pair(iterator, mask);
}

template <typename Key, typename Value, typename Hash>
__device__ uint8_t
SlabHashContext<Key, Value, Hash>::Remove(uint8_t& to_be_deleted,
                                          const uint32_t lane_id,
                                          const uint32_t bucket_id,
                                          const Key& key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    uint8_t mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_deleted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        Key src_key;
        WarpSyncKey(key, src_lane, src_key);

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            ptr_t src_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_found)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_found);
                ptr_t pair_to_delete = *unit_data_ptr;

                // TODO: keep in mind the potential double free problem
                ptr_t old_key_value_pair =
                        atomicCAS((unsigned int*)(unit_data_ptr),
                                  pair_to_delete, EMPTY_PAIR_PTR);
                /** Branch 1.1: this thread reset, free src_addr **/
                if (old_key_value_pair == pair_to_delete) {
                    pair_allocator_ctx_.Free(src_pair_internal_ptr);
                    mask = true;
                }
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
                to_be_deleted = false;
            }
        } else {  // no matching slot found:
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                // not found:
                to_be_deleted = false;
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }

    return mask;
}

template <typename Key, typename Value, typename Hash>
__global__ void SearchKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* keys,
                             _Iterator<Key, Value>* iterators,
                             uint8_t* masks,
                             uint32_t num_queries) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= num_queries) {
        return;
    }

    /* Initialize the memory allocator on each warp */
    slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    Key key;

    if (tid < num_queries) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    _Pair<ptr_t, uint8_t> result =
            slab_hash_ctx.Search(lane_active, lane_id, bucket_id, key);

    if (tid < num_queries) {
        iterators[tid] = slab_hash_ctx.get_pair_alloc_ctx().extract_ext_ptr(
                result.first);
        masks[tid] = result.second;
    }
}

template <typename Key, typename Value, typename Hash>
__global__ void InsertKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* keys,
                             Value* values,
                             _Iterator<Key, Value>* iterators,
                             uint8_t* masks,
                             uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    Key key;
    Value value;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        value = values[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    _Pair<ptr_t, uint8_t> result =
            slab_hash_ctx.Insert(lane_active, lane_id, bucket_id, key, value);

    if (tid < num_keys) {
        iterators[tid] = slab_hash_ctx.get_pair_alloc_ctx().extract_ext_ptr(
                result.first);
        masks[tid] = result.second;
    }
}

template <typename Key, typename Value, typename Hash>
__global__ void RemoveKernel(SlabHashContext<Key, Value, Hash> slab_hash_ctx,
                             Key* keys,
                             uint8_t* masks,
                             uint32_t num_keys) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if ((tid - lane_id) >= num_keys) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id);

    uint8_t lane_active = false;
    uint32_t bucket_id = 0;
    Key key;

    if (tid < num_keys) {
        lane_active = true;
        key = keys[tid];
        bucket_id = slab_hash_ctx.ComputeBucket(key);
    }

    uint8_t success =
            slab_hash_ctx.Remove(lane_active, lane_id, bucket_id, key);

    if (tid < num_keys) {
        masks[tid] = success;
    }
}

template <typename Key, typename Value, typename Hash>
__global__ void GetIteratorsKernel(
        SlabHashContext<Key, Value, Hash> slab_hash_ctx,
        ptr_t* iterators,
        uint32_t* iterator_count,
        uint32_t num_buckets) {
    // global warp ID
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wid = tid >> 5;
    // assigning a warp per bucket
    if (wid >= num_buckets) {
        return;
    }

    /* uint32_t lane_id = threadIdx.x & 0x1F; */

    /* // initializing the memory allocator on each warp: */
    /* slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id); */

    /* uint32_t src_unit_data = */
    /*         *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id); */
    /* uint32_t active_mask = */
    /*         __ballot_sync(PAIR_PTR_LANES_MASK, src_unit_data !=
     * EMPTY_PAIR_PTR); */
    /* int leader = __ffs(active_mask) - 1; */
    /* uint32_t count = __popc(active_mask); */
    /* uint32_t rank = __popc(active_mask & __lanemask_lt()); */
    /* uint32_t prev_count; */
    /* if (rank == 0) { */
    /*     prev_count = atomicAdd(iterator_count, count); */
    /* } */
    /* prev_count = __shfl_sync(active_mask, prev_count, leader); */

    /* if (src_unit_data != EMPTY_PAIR_PTR) { */
    /*     iterators[prev_count + rank] = src_unit_data; */
    /* } */

    /* uint32_t next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* while (next != EMPTY_SLAB_PTR) { */
    /*     src_unit_data = */
    /*             *slab_hash_ctx.get_unit_ptr_from_list_nodes(next,
     * lane_id);
     */
    /*     count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK, */
    /*                                   src_unit_data != EMPTY_PAIR_PTR));
     */
    /*     next = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32); */
    /* } */
    /* // writing back the results: */
    /* if (lane_id == 0) { */
    /* } */
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename Key, typename Value, typename Hash>
__global__ void CountElemsPerBucketKernel(
        SlabHashContext<Key, Value, Hash> slab_hash_ctx,
        uint32_t* bucket_elem_counts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    uint32_t wid = tid >> 5;
    if (wid >= slab_hash_ctx.bucket_size()) {
        return;
    }

    slab_hash_ctx.get_slab_alloc_ctx().Init(tid, lane_id);

    uint32_t count = 0;

    // count head node
    uint32_t src_unit_data =
            *slab_hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                  src_unit_data != EMPTY_PAIR_PTR));
    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data =
                *slab_hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                      src_unit_data != EMPTY_PAIR_PTR));
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }

    // write back the results:
    if (lane_id == 0) {
        bucket_elem_counts[wid] = count;
    }
}
