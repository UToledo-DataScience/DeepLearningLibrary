#ifndef TENSOR
#define TENSOR
#include <cassert>
#include <vector>
#include <memory>
#include "core/buffer.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

template <typename TensorDType>
class Tensor {
    uint32_t children;
    std::vector<int>* shape;

    // questioning the need for shared_ptr
    Buffer<TensorDType>* backend;
    Allocator<TensorDType>* allocator;

    Operation<TensorDType>* operation;

    Buffer<TensorDType>* getBackend() { return backend; }

    void incrChildren() { children++; }

  public:
    // fresh tensor
    Tensor(std::vector<TensorDType> values, std::vector<int> s, Allocator<TensorDType>* a);

    // tensor from binary operation
    // TODO: immediate allocation NEEDS to be changed to memory being allocated
    //       at a later time e.g. when the user calls Tensor.operate()
    Tensor(Tensor<TensorDType>* t1, Tensor<TensorDType>* t2, Operation<TensorDType>* op);

    // TODO: REFERENCE COUNTS
    ~Tensor();

    // operates the tensor,
    // bringing the data in the buffer up to speed
    // at the current operation
    void operate();

    std::vector<int>& getShape();

    uint64_t getSize();

    uint32_t getChildren();

    Operation<TensorDType>* getOperation();

    Allocator<TensorDType>* getAllocator();

    Buffer<TensorDType>* getBuffer();

    void setBuffer(Buffer<TensorDType>* buf);

    void print();
};

} // namespace deeplib

#include "core/tensor.cpp"
#endif
