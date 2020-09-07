#ifndef TENSOR
#define TENSOR
#include <cassert>
#include <vector>
#include <memory>
#include "core/buffer.h"
#include "core/operations.h"
#include "core/data_types.h"

namespace deeplib {

class Tensor {
    uint32_t children;
    std::vector<int>* shape;

    Buffer* buffer;
    Allocator* allocator;

    DataType dtype;

    Operation* operation;

    void incrChildren() { children++; }

  public:
    Tensor(std::vector<int> newShape, DataType dt, Allocator* a);

    // fresh tensor
    // NOTE: This will have to be changed.
    //       Tensors initialized from a set of values will have to happen
    //       some other way.
    Tensor(std::vector<int> values, std::vector<int> s, Allocator* a);

    // tensor from binary operation
    // TODO: immediate allocation NEEDS to be changed to memory being allocated
    //       at a later time e.g. when the user calls Tensor.operate()
    Tensor(Tensor* t1, Tensor* t2, Operation* op);

    // TODO: REFERENCE COUNTS
    ~Tensor();

    // operates the tensor,
    // bringing the data in the buffer up to speed
    // at the current operation
    void operate();

    std::vector<int>& getShape();

    uint64_t getSize();

    uint32_t getChildren();

    Operation* getOperation();

    DataType getDataType();

    Allocator* getAllocator();

    Buffer* getBuffer();

    void setBuffer(Buffer* buf);

    void print();
};

} // namespace deeplib

#endif
