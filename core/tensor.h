#ifndef TENSOR
#define TENSOR
#include <cassert>
#include <vector>
#include <memory>
#include "core/buffer.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

class Operation;

template <typename TensorDType>
class Tensor {
    uint32_t children;
    std::vector<int>* shape;

    // questioning the need for shared_ptr
    std::shared_ptr<Buffer<TensorDType>> backend;
    Allocator<TensorDType>* allocator;

    operations::Operation<TensorDType>* operation;

    std::shared_ptr<Buffer<TensorDType>> getBackend() { return backend; }

    void incrChildren() { children++; }

  public:
    // fresh tensor
    Tensor(std::vector<TensorDType> values, std::vector<int> s, Allocator<TensorDType>* a) {
        children = 0;
        allocator = a;
        backend = std::make_shared<Buffer<TensorDType>>(Buffer<TensorDType>(values, s, a));
        operation = new operations::Constant<TensorDType>(backend);
    }

    // tensor from binary operation
    // TODO: immediate allocation NEEDS to be changed to memory being allocated
    //       at a later time e.g. when the user calls Tensor.operate()
    Tensor(Tensor<TensorDType>* t1, Tensor<TensorDType>* t2, operations::Operation<TensorDType>* op) {
        children = 0;

        if (t1 == t2) {
            backend = std::make_shared<Buffer<TensorDType>>(
                        Buffer<TensorDType>(t1->getShape(), t1->getAllocator()));

            t1->incrChildren();
        }

        // Buffer shenanigans if a tensor is used in two or more operations.
        //
        // For each tensor, n-1 buffers need to be allocated
        // where n is the number of children from the tensor's operation node.
        //
        // When a new buffer is allocated, the initially allocated buffer
        // needs moved to the latest tensor so that in-place calculations
        // don't overwrite the original buffer and throw off the rest
        // of the calculation graph.
        else if (t1->getSize() >= t2->getSize()) {
            if (t1->getChildren() > 0) {
                backend = t1->getBuffer();
                t1->setBuffer(std::make_shared<Buffer<TensorDType>>(
                        Buffer<TensorDType>(t1->getBuffer())));

                t1->incrChildren();
            }
            // to also account for if t1.size == t2.size
            else if (t2->getChildren() > 0) {
                backend = t2->getBuffer();
                t2->setBuffer(std::make_shared<Buffer<TensorDType>>(
                        Buffer<TensorDType>(t2->getBuffer())));

                t2->incrChildren();
            }
            else {
                backend = t1->getBackend();
                t1->incrChildren();
            }
        }
        else {
            if (t2->getChildren() > 0) {
                backend = t1->getBuffer();
                t2->setBuffer(std::make_shared<Buffer<TensorDType>>(
                        Buffer<TensorDType>(t2->getBuffer())));

                t2->incrChildren();
            }
            else {
                backend = t2->getBackend();
                t2->incrChildren();
            }
        }

        operation = op;

        operation->setBuffer(backend);
    }

    // TODO: REFERENCE COUNTS
    ~Tensor() {
        //delete placeholder;
        //delete operation;
    }

    // operates the tensor,
    // bringing the data in the buffer up to speed
    // at the current operation
    void operate() { operation->operate(); }

    std::vector<int>& getShape() { return backend->getShape(); }

    uint64_t getSize() { return backend->getSize(); }

    uint32_t getChildren() { return children; }

    operations::Operation<TensorDType>* getOperation() { return operation; }

    Allocator<TensorDType>* getAllocator() { return backend->getAllocator(); }

    std::shared_ptr<Buffer<TensorDType>> getBuffer() { return backend; }

    void setBuffer(std::shared_ptr<Buffer<TensorDType>> buf) {
        backend = buf;
        operation->setBuffer(buf);
    }

    void print() {
        backend->print();
    }
};

} // namespace deeplib
#endif
