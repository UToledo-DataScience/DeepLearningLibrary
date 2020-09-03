#include <cassert>
#include <vector>
#include <memory>
#include "core/tensor.h"
#include "core/buffer.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

template<typename TensorDType>
Tensor<TensorDType>::Tensor(std::vector<TensorDType> values, std::vector<int> s, Allocator<TensorDType>* a) {
    children = 0;
    allocator = a;
    backend = allocator->newBuffer(new Buffer<TensorDType>(values, s, a));
    operation = allocator->newOperation(new Constant<TensorDType>(backend));
}

// Tensor from a binary operation.
//
// NOTE: Since the tensor resulting from this operation will
//       inherit the allocator of its parents, this objects
//
// TODO: immediate allocation NEEDS to be changed to memory being allocated
//       at a later time e.g. when the user calls Tensor.operate()
template <typename TensorDType>
Tensor<TensorDType>::Tensor(Tensor<TensorDType>* t1, Tensor<TensorDType>* t2, Operation<TensorDType>* op) {
    children = 0;

    if (t1->getAllocator() == t2->getAllocator())
        allocator = t1->getAllocator();
    else {
        std::cout << "Error: allocator mismatch in tensor instantiation. "
                  << "This should not be happening" << std::endl;

        exit(1);
    }

    if (t1 == t2) {
        backend = allocator->newBuffer(new Buffer<TensorDType>(t1->getShape(), t1->getAllocator()));

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
            t1->setBuffer(allocator->newBuffer(new Buffer<TensorDType>(t1->getBuffer())));

            t1->incrChildren();
        }
        // to also account for if t1.size == t2.size
        else if (t2->getChildren() > 0) {
            backend = t2->getBuffer();
            t2->setBuffer(allocator->newBuffer(new Buffer<TensorDType>(t2->getBuffer())));

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
            t2->setBuffer(allocator->newBuffer(new Buffer<TensorDType>(t2->getBuffer())));

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
template <typename TensorDType>
Tensor<TensorDType>::~Tensor() {
    //delete placeholder;
    //delete operation;
}

// operates the tensor,
// bringing the data in the buffer up to speed
// at the current operation
template <typename TensorDType>
void Tensor<TensorDType>::operate() { operation->operate(); }

template <typename TensorDType>
std::vector<int>& Tensor<TensorDType>::getShape() { return backend->getShape(); }

template <typename TensorDType>
uint64_t Tensor<TensorDType>::getSize() { return backend->getSize(); }

template <typename TensorDType>
uint32_t Tensor<TensorDType>::getChildren() { return children; }

template <typename TensorDType>
Operation<TensorDType>* Tensor<TensorDType>::getOperation() { return operation; }

template <typename TensorDType>
Allocator<TensorDType>* Tensor<TensorDType>::getAllocator() { return backend->getAllocator(); }

template <typename TensorDType>
Buffer<TensorDType>* Tensor<TensorDType>::getBuffer() { return backend; }

template <typename TensorDType>
void Tensor<TensorDType>::setBuffer(Buffer<TensorDType>* buf) {
    backend = buf;
    operation->setBuffer(buf);
}

template <typename TensorDType>
void Tensor<TensorDType>::print() {
    backend->print();
}

} // namespace deeplib
