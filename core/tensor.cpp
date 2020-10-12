#include <cassert>
#include <vector>
#include <memory>
#include "core/data_types.h"
#include "core/tensor.h"
#include "core/buffer.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

Tensor::Tensor(std::vector<int> new_shape, DataType data_type, Allocator* a) {
    children_ = 0;
    allocator_ = a;

    dtype_ = data_type;

    buffer_ = allocator_->newBuffer(new Buffer(new_shape, a));
    operation_ = allocator_->newOperation(new Constant(buffer_)); // ?? subject to change
}

Tensor::Tensor(std::vector<int> values, std::vector<int> shape, Allocator* a) {
    children_ = 0;
    allocator_ = a;

    dtype_ = DataType::INT32;

    buffer_ = allocator_->newBuffer(new Buffer(values, shape, a));
    operation_ = allocator_->newOperation(new Constant(buffer_));
}

Tensor::Tensor(Tensor& t1, Tensor& t2, Operation* op) {
    children_ = 0;

    // NOTE: Since the tensor resulting from this operation will
    //       inherit the allocator of its parents, this objects
    if (t1.getAllocator() == t2.getAllocator())
        allocator_ = t1.getAllocator();
    else {
        std::cout << "Error: allocator mismatch in tensor instantiation. "
                  << "This should not be happening" << std::endl;

        exit(1);
    }

    if (&t1 == &t2) {
        buffer_ = allocator_->newBuffer(new Buffer(t1.getShape(), t1.getAllocator()));

        t1.incrChildren();
    }

    // Buffer shenanigans if a single tensor is used in two or more operations.
    //
    // For each tensor, n-1 buffers need to be allocated
    // where n is the number of children from the tensor's operation node.
    //
    // When a new buffer is allocated, the initially allocated buffer
    // needs moved to the latest tensor so that in-place calculations
    // don't overwrite the original buffer and throw off the rest
    // of the calculation graph.
    else if (t1.getSize() >= t2.getSize()) {
        if (t1.getChildren() > 0) {
            buffer_ = t1.getBuffer();
            t1.setBuffer(allocator_->newBuffer(new Buffer(t1.getBuffer())));

            t1.incrChildren();
        }
        // to also account for if t1.size == t2.size
        else if (t2.getChildren() > 0) {
            buffer_ = t2.getBuffer();
            t2.setBuffer(allocator_->newBuffer(new Buffer(t2.getBuffer())));

            t2.incrChildren();
        }
        else {
            buffer_ = t1.getBuffer();
            t1.incrChildren();
        }
    }
    else {
        if (t2.getChildren() > 0) {
            buffer_ = t1.getBuffer();
            t2.setBuffer(allocator_->newBuffer(new Buffer(t2.getBuffer())));

            t2.incrChildren();
        }
        else {
            buffer_ = t2.getBuffer();
            t2.incrChildren();
        }
    }

    // data type checks should have been performed by now
    dtype_ = t1.getDataType();
    operation_ = op;
    operation_->setBuffer(buffer_);
}

Tensor::Tensor(Tensor& t1, Tensor& t2, Operation* op, std::vector<int> new_shape) {
    children_ = 0;

    allocator_ = t1.getAllocator();
    buffer_ = allocator_->newBuffer(new Buffer(new_shape, allocator_));
    dtype_ = t1.getDataType();
    operation_ = op;
    operation_->setBuffer(buffer_);
}

Tensor::Tensor(Tensor& t, Operation* op) {
    children_ = 0;
    t.incrChildren();

    allocator_ = t.getAllocator();
    buffer_ = t.getBuffer();
    dtype_ = t.getDataType();
    operation_ = op;
    operation_->setBuffer(buffer_);
}

Tensor::Tensor(Tensor& t, Operation* op, DataType new_dtype) {
    children_ = 0;
    t.incrChildren();

    allocator_ = t.getAllocator();

    // TODO: Check to see if previous buffer is smaller or larger in data type size.
    //if (t.getBuffer()->getDataType() < )
    buffer_ = allocator_->newBuffer(new Buffer(t.getBuffer(), new_dtype));
    //else
    //    buffer_ = t.getBuffer();

    dtype_ = new_dtype;
    operation_ = op;
    operation_->setBuffer(buffer_);
}

Tensor::~Tensor() {}

void Tensor::operate() {
    operation_->operate();
}

void Tensor::uproot() {
    allocator_->uprootOperation(operation_);
}

std::vector<int>& Tensor::getShape() {
    return buffer_->getShape();
}

uint64_t Tensor::getSize() {
    return buffer_->getSize();
}

DataType Tensor::getDataType() {
    return dtype_;
}

uint32_t Tensor::getChildren() {
    return children_;
}

Operation* Tensor::getOperation() {
    return operation_;
}

Allocator* Tensor::getAllocator() {
    return buffer_->getAllocator();
}

Buffer* Tensor::getBuffer() {
    return buffer_;
}

void Tensor::setBuffer(Buffer* buf) {
    buffer_ = buf;
    operation_->setBuffer(buf);
}

void Tensor::print() {
    switch (dtype_) {
      case DataType::UINT8:
        buffer_->print<uint8_t>();
        return;

      case DataType::UINT16:
        buffer_->print<uint16_t>();
        return;

      case DataType::UINT32:
        buffer_->print<uint32_t>();
        return;

      case DataType::UINT64:
        buffer_->print<uint64_t>();
        return;

      case DataType::INT8:
        buffer_->print<int8_t>();
        return;

      case DataType::INT16:
        buffer_->print<int16_t>();
        return;

      case DataType::INT32:
        buffer_->print<int32_t>();
        return;

      case DataType::INT64:
        buffer_->print<int64_t>();
        return;
        
      case DataType::FLOAT32:
        buffer_->print<float>();
        return;

      case DataType::FLOAT64:
        buffer_->print<double>();
        return;

      case DataType::BOOL:
        buffer_->print<bool>();
        return;

      default:
        std::cout << "ERROR: Bad data type!" << std::endl;
        assert(false);
    }
}

} // namespace deeplib
