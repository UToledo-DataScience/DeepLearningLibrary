#ifndef PLACEHOLDER
#define PLACEHOLDER
#include <iostream>
#include <cassert>

// class for allocating data
// TODO: ensure it outlives the tensors it allocates
template <class AlDType>
class Allocator {
    uint32_t total_allocations;
    uint64_t bytes_allocated;
    uint64_t bytes_unallocated;
    uint64_t bytes_currently_allocated;

  public:
    Allocator(): total_allocations(0),
    bytes_allocated(0),
    bytes_unallocated(0),
    bytes_currently_allocated(0)
    {}

    AlDType* allocate(uint64_t count) {
        AlDType* data = (AlDType*)calloc(count, sizeof(AlDType));

        bytes_allocated += count * sizeof(AlDType);
        bytes_currently_allocated = bytes_allocated;
        total_allocations++;

        return data;
    }

    AlDType* deallocate(AlDType* data) {
        free(data);
        data = nullptr;

        bytes_unallocated += bytes_currently_allocated;
        bytes_currently_allocated = 0;

        return data;
    }

    AlDType* reallocate(AlDType* data, uint64_t new_count) {
        free(data);
        data = (AlDType*)calloc(new_count, sizeof(AlDType));

        bytes_unallocated += bytes_currently_allocated;
        bytes_currently_allocated = new_count * sizeof(AlDType);
        bytes_allocated += bytes_currently_allocated;
        total_allocations++;

        return data;
    }

    void printStats() {
        std::cout << "total_allocations: " << total_allocations << std::endl
                  << "bytes_allocated: " << bytes_allocated << std::endl
                  << "bytes_unallocated: " << bytes_unallocated << std::endl
                  << "bytes_currently_allocated: " << bytes_currently_allocated << std::endl;
    }
};

#endif
