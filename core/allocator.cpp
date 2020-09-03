#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// move this
template <typename T>
bool in(T* op, std::vector<T*> vec) {
    for (auto i : vec) {
        if (op == i)
            return true;
    }

    return false;
}

template<typename AlDType>
Allocator<AlDType>::Allocator(): total_allocations(0),
                                 total_deallocations(0),
                                 bytes_allocated(0),
                                 bytes_deallocated(0),
                                 bytes_currently_allocated(0)
    {}

// The following two functions are wrapper functions
// for newly instantiated operations and buffers
// used to keep track of what's allocated by this allocator.
//
// i.e. new Operation/Buffer shouldn't be used without these
//
// How to use:
//   X<...>* example = newX(new X<...>(...))
//   where X is either Operation or Buffer

template<typename AlDType>
Operation<AlDType>* Allocator<AlDType>::newOperation(Operation<AlDType>* new_op) {
    operations.push_back(new_op);

    bytes_allocated += sizeof(Operation<AlDType>);
    bytes_currently_allocated += sizeof(Operation<AlDType>);
    total_allocations++;

    return new_op;
}

template<typename AlDType>
Buffer<AlDType>* Allocator<AlDType>::newBuffer(Buffer<AlDType>* new_buf) {
    buffers.push_back(new_buf);

    bytes_allocated += sizeof(Buffer<AlDType>);
    bytes_currently_allocated += sizeof(Buffer<AlDType>);
    total_allocations++;

    return new_buf;
}

template<typename AlDType>
AlDType* Allocator<AlDType>::allocate(uint64_t count) {
    AlDType* data = (AlDType*)calloc(count, sizeof(AlDType));

    uint64_t newly_allocated = count * sizeof(AlDType);
    bytes_allocated += newly_allocated;
    bytes_currently_allocated += newly_allocated;

    return data;
}

template<typename AlDType>
void Allocator<AlDType>::freeBuffer(Buffer<AlDType>* buf) {
    if (buf->buffer_data != nullptr) {
        free(buf->buffer_data);
        buf->buffer_data = nullptr;

        delete buf;

        uint64_t dealloc_size = buf->total_size * sizeof(AlDType) + sizeof(Buffer<AlDType>);

        bytes_deallocated += dealloc_size;
        bytes_currently_allocated -= dealloc_size;
        total_deallocations++;
    }
}

template<typename AlDType>
AlDType* Allocator<AlDType>::reallocate(AlDType* data, uint64_t new_count) {
    free(data);
    data = (AlDType*)calloc(new_count, sizeof(AlDType));

    bytes_deallocated += bytes_currently_allocated;
    bytes_currently_allocated = new_count * sizeof(AlDType);
    bytes_allocated += bytes_currently_allocated;
    total_allocations++;

    return data;
}

// Cleanup function. This deallocates the given tensor
// as well as ALL of its ancestors.
// This includes freeing the Operation and Buffer pointers
// associated with the tensor in the graph.
//
// Tensors that have been uprooted cannot be used again
// or errors will result. TODO: error handling
template<typename AlDType>
void Allocator<AlDType>::uproot(Tensor<AlDType>* tensor) {
    for (auto buf : buffers)
        freeBuffer(buf);

    for (auto oper : operations) {
        delete oper;
        bytes_deallocated += sizeof(Operation<AlDType>);
        bytes_currently_allocated -= sizeof(Operation<AlDType>);
        total_deallocations++;
    }
}

template<typename AlDType>
void Allocator<AlDType>::printStats() {
    std::cout << "total_allocations: " << total_allocations << std::endl
              << "total_deallocations: " << total_deallocations << std::endl
              << "bytes_allocated: " << bytes_allocated << std::endl
              << "bytes_deallocated: " << bytes_deallocated << std::endl
              << "bytes_currently_allocated: " << bytes_currently_allocated << std::endl;
}

} // namespace deeplib
