#include <cassert>
#include <vector>
#include <memory>
#include "core/data_types.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

Tensor::Tensor(std::vector<int> newShape, DataType data_type, Allocator* a) {
    children = 0;
    allocator = a;

    dtype = dtype;

    buffer = allocator->newBuffer(new Buffer(newShape, a));
    operation = allocator->newOperation(new Constant(buffer)); // ?? subject to change
}

Tensor::Tensor(std::vector<int> values, std::vector<int> newShape, Allocator* a) {
    children = 0;
    allocator = a;

    dtype = DataType::INT32;

    buffer = allocator->newBuffer(new Buffer(values, newShape, a));
    operation = allocator->newOperation(new Constant(buffer));
}

// Tensor from a binary operation.
//
// NOTE: Since the tensor resulting from this operation will
//       inherit the allocator of its parents, this objects
//
// TODO: immediate allocation NEEDS to be changed to memory being allocated
//       at a later time e.g. when the user calls Tensor.operate()
Tensor::Tensor(Tensor* t1, Tensor* t2, Operation* op) {
    children = 0;

    if (t1->getAllocator() == t2->getAllocator())
        allocator = t1->getAllocator();
    else {
        std::cout << "Error: allocator mismatch in tensor instantiation. "
                  << "This should not be happening" << std::endl;

        exit(1);
    }

    if (t1 == t2) {
        buffer = allocator->newBuffer(new Buffer(t1->getShape(), t1->getAllocator()));

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
            buffer = t1->getBuffer();
            t1->setBuffer(allocator->newBuffer(new Buffer(t1->getBuffer())));

            t1->incrChildren();
        }
        // to also account for if t1.size == t2.size
        else if (t2->getChildren() > 0) {
            buffer = t2->getBuffer();
            t2->setBuffer(allocator->newBuffer(new Buffer(t2->getBuffer())));

            t2->incrChildren();
        }
        else {
            buffer = t1->getBuffer();
            t1->incrChildren();
        }
    }
    else {
        if (t2->getChildren() > 0) {
            buffer = t1->getBuffer();
            t2->setBuffer(allocator->newBuffer(new Buffer(t2->getBuffer())));

            t2->incrChildren();
        }
        else {
            buffer = t2->getBuffer();
            t2->incrChildren();
        }
    }

    // data type checks should have been performed by now
    dtype = t1->getDataType();

    operation = op;

    operation->setBuffer(buffer);
}

Tensor::~Tensor() {}

// operates the tensor,
// bringing the data in the buffer up to speed
// at the current operation
void Tensor::operate() {
    operation->operate();
}

std::vector<int>& Tensor::getShape() { return buffer->getShape(); }

uint64_t Tensor::getSize() { return buffer->getSize(); }

DataType Tensor::getDataType() { return dtype; }

uint32_t Tensor::getChildren() { return children; }

Operation* Tensor::getOperation() { return operation; }

Allocator* Tensor::getAllocator() { return buffer->getAllocator(); }

Buffer* Tensor::getBuffer() { return buffer; }

void Tensor::setBuffer(Buffer* buf) {
    buffer = buf;
    operation->setBuffer(buf);
}

void Tensor::print() {
    switch (dtype) {
      case DataType::UINT8:
        buffer->print<uint8_t>();
        return;

      case DataType::UINT16:
        buffer->print<uint16_t>();
        return;

      case DataType::UINT32:
        buffer->print<uint32_t>();
        return;

      case DataType::UINT64:
        buffer->print<uint64_t>();
        return;

      case DataType::INT8:
        buffer->print<int8_t>();
        return;

      case DataType::INT16:
        buffer->print<int16_t>();
        return;

      case DataType::INT32:
        buffer->print<int32_t>();
        return;

      case DataType::INT64:
        buffer->print<int64_t>();
        return;
        
      case DataType::FLOAT32:
        buffer->print<float>();
        return;

      case DataType::FLOAT64:
        buffer->print<double>();
        return;

      case DataType::BOOL:
        buffer->print<bool>();
        return;

      default:
        std::cout << "ERROR: Bad data type!" << std::endl;
        assert(false);
    }
}

} // namespace deeplib
