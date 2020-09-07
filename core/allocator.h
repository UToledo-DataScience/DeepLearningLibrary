#ifndef PLACEHOLDER
#define PLACEHOLDER
#include <iostream>
#include <cassert>

namespace deeplib {

class Tensor;
class Operation;
class Buffer;

// Container for handling memory allocation and cleanup.
// Keeps track of the operations and buffers allocated 
// within a single graph and single data type.
//
// Once this object is destroyed, it deallocates everything
// under it's watch. Thus all tensors created using this
// can never be used outside the scope of this Allocator.
class Allocator {
    uint32_t total_allocations_;
    uint32_t total_deallocations_;
    uint64_t bytes_allocated_;
    uint64_t bytes_deallocated_;
    uint64_t bytes_currently_allocated_;

    std::vector<Operation*> operations_;
    std::vector<Buffer*> buffers_;

  public:
    Allocator();

    ~Allocator();

    // NOTE: The following new functions MUST be used whenver
    //       `new Operation(...)` or `new Buffer(...)` is written.
    //       Otherwise, the allocated objects will likely
    //       (though not certainly) leak.

    // Register a new operation under this allocator.
    Operation* newOperation(Operation* new_op);

    // Register a new buffer under this allocator.
    Buffer* newBuffer(Buffer* new_buf);

    // Allocates `count` elements of data type `AlDType`.
    template <typename AlDType>
    void* allocate(uint64_t count);

    // Deallocates the given buffer, rendering it unusable.
    void freeBuffer(Buffer* buf);

    // Deallocates EVERYTHING allocated by this allocator.
    void uproot();

    // Child function.
    void uprootOperation(Operation* op, int& index);

    // Deallocate the buffers and operations of ALL the ancestors
    // of the given operation.
    //
    // TODO: Put more thought into this function's construction.
    //       Doesn't quite feel like it's robust enough for general use.
    void uprootOperation(Operation* op);

    // Give a quick summary of everything this Allocator has allocated
    // and deallocated.
    void printStats();
};

} // namespace deeplib

#include "core/allocator.t.h"
#endif
