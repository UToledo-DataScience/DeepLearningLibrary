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
// NOTE: the following is more or less planning.
//       Nothing is yet implemented and the plan isn't yet solid
//       so it is currently (9-2-20) subject to change.
//
// Should data types change or two graphs merge,
// a new child allocator is created which handles everything onward.
class Allocator {
    uint32_t total_allocations;
    uint32_t total_deallocations;
    uint64_t bytes_allocated;
    uint64_t bytes_deallocated;
    uint64_t bytes_currently_allocated;

    std::vector<Operation*> operations;
    std::vector<Buffer*> buffers;

  public:
    Allocator();

    Operation* newOperation(Operation* new_op);

    Buffer* newBuffer(Buffer* new_buf);

    template <typename AlDType>
    void* allocate(uint64_t count);

    void freeBuffer(Buffer* buf);

    void uproot(Tensor* tensor);

    void printStats();
};

} // namespace deeplib

#include "core/allocator.t.h"
#endif
