#ifndef PLACEHOLDER
#define PLACEHOLDER
#include <iostream>
#include <cassert>
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

template <class T>
class Tensor;

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
template <class AlDType>
class Allocator {
    uint32_t total_allocations;
    uint32_t total_deallocations;
    uint64_t bytes_allocated;
    uint64_t bytes_deallocated;
    uint64_t bytes_currently_allocated;

    std::vector<Operation<AlDType>*> operations;
    std::vector<Buffer<AlDType>*> buffers;

  public:
    Allocator();

    Operation<AlDType>* newOperation(Operation<AlDType>* new_op);

    Buffer<AlDType>* newBuffer(Buffer<AlDType>* new_buf);

    AlDType* allocate(uint64_t count);

    void freeBuffer(Buffer<AlDType>* buf);

    AlDType* reallocate(AlDType* data, uint64_t new_count);

    void uproot(Tensor<AlDType>* tensor);

    void printStats();
};

} // namespace deeplib

#include "core/allocator.cpp"
#endif
