#include <iostream>
#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/operations.h"
#include "core/buffer.h"

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

Allocator::Allocator(): total_allocations(0),
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

Operation* Allocator::newOperation(Operation* new_op) {
    operations.push_back(new_op);

    bytes_allocated += sizeof(Operation);
    bytes_currently_allocated += sizeof(Operation);
    total_allocations++;

    return new_op;
}

Buffer* Allocator::newBuffer(Buffer* new_buf) {
    buffers.push_back(new_buf);

    bytes_allocated += sizeof(Buffer);
    bytes_currently_allocated += sizeof(Buffer);
    total_allocations++;

    return new_buf;
}

void Allocator::freeBuffer(Buffer* buf) {
    if (buf->buffer_data != nullptr) {
        free(buf->buffer_data);
        buf->buffer_data = nullptr;

        delete buf;

        uint64_t dealloc_size = buf->total_size + sizeof(Buffer);

        bytes_deallocated += dealloc_size;
        bytes_currently_allocated -= dealloc_size;
        total_deallocations++;
    }
}

// Cleanup function. This deallocates the given tensor
// as well as ALL of its ancestors.
// This includes freeing the Operation and Buffer pointers
// associated with the tensor in the graph.
//
// Tensors that have been uprooted cannot be used again
// or errors will result. TODO: error handling
void Allocator::uproot(Tensor* tensor) {
    for (auto buf : buffers)
        freeBuffer(buf);

    for (auto oper : operations) {
        delete oper;
        bytes_deallocated += sizeof(Operation);
        bytes_currently_allocated -= sizeof(Operation);
        total_deallocations++;
    }
}

void Allocator::printStats() {
    std::cout << "total_allocations: " << total_allocations << std::endl
              << "total_deallocations: " << total_deallocations << std::endl
              << "bytes_allocated: " << bytes_allocated << std::endl
              << "bytes_deallocated: " << bytes_deallocated << std::endl
              << "bytes_currently_allocated: " << bytes_currently_allocated << std::endl;
}

} // namespace deeplib
