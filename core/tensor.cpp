#include <stack>
#include <vector>
#include <memory>
#include <cassert>
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

Tensor::Tensor(Tensor& t) {
    children_ = t.getChildren();
    allocator_ = t.getAllocator();
    buffer_ = t.getBuffer();
    dtype_ = t.getDataType();
    operation_ = t.getOperation();
}

Tensor::~Tensor() {}

// Helper for Tensor::operate()
bool Tensor::isConstant(Operation* op) {
    return !op->type_.compare("constant");
}

bool Tensor::isNary(Operation* op, int n) {
    return op->ary_ == n;
}

// NOTE: Only supports up to binary operations.
// This should probably be moved.
void Tensor::operate() {
    if (isConstant(this->operation_)) {
        this->operation_->operate();
        return;
    }
    else {
        std::stack<Operation*> operation_buffer1;
        std::stack<Operation*> operation_buffer2;
        std::stack<Operation*> computed_operations;
        Operation* op = this->operation_;
        operation_buffer1.push(op);
        while (operation_buffer1.size() > 0) {
            op = operation_buffer1.top();
            operation_buffer2.push(op);
            operation_buffer1.pop();

            if (isNary(op, 2)) {
                if (op->parent1_)
                    operation_buffer1.push(op->parent1_);

                if (op->parent2_)
                    operation_buffer1.push(op->parent2_);
            }
            else {
                if (op->parent1_)
                    operation_buffer1.push(op->parent1_);
            }
        }

        // This can probably be cleaned up.
        while (operation_buffer2.size() > 0) {
            Operation* buf1_top = operation_buffer1.size() > 0 ? operation_buffer1.top() : 0;
            Operation* buf2_top = operation_buffer2.top();
            Operation* comp_top = computed_operations.size() > 0 ? computed_operations.top() : 0;
            if (isConstant(buf2_top)) {
                computed_operations.push(buf2_top);
                operation_buffer2.pop();
            }
            else if (comp_top) {
                if (isNary(buf2_top, 1) && (isConstant(comp_top) || comp_top->computed_)) {
                    buf2_top->operate(comp_top->buffer_);
                    computed_operations.pop();
                    computed_operations.push(buf2_top);
                    operation_buffer2.pop();

                    continue;
                }
                if (buf2_top) {
                    if (isNary(buf2_top, 2) && computed_operations.size() > 1) {
                        Operation* op1 = computed_operations.top();
                        computed_operations.pop();
                        Operation* op2 = computed_operations.top();
                        computed_operations.pop();

                        buf2_top->operate(op2->buffer_, op1->buffer_);

                        computed_operations.push(buf2_top);
                        operation_buffer2.pop();

                        continue;
                    }
                }
            }
        }
    }
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

void Tensor::setName(std::string name) {
    operation_->setName(name);
}

void Tensor::print(bool linear) {
    switch (dtype_) {
      case DataType::UINT8:
        buffer_->print<uint8_t>(linear);
        return;

      case DataType::UINT16:
        buffer_->print<uint16_t>(linear);
        return;

      case DataType::UINT32:
        buffer_->print<uint32_t>(linear);
        return;

      case DataType::UINT64:
        buffer_->print<uint64_t>(linear);
        return;

      case DataType::INT8:
        buffer_->print<int8_t>(linear);
        return;

      case DataType::INT16:
        buffer_->print<int16_t>(linear);
        return;

      case DataType::INT32:
        buffer_->print<int32_t>(linear);
        return;

      case DataType::INT64:
        buffer_->print<int64_t>(linear);
        return;
        
      case DataType::FLOAT32:
        buffer_->print<float>(linear);
        return;

      case DataType::FLOAT64:
        buffer_->print<double>(linear);
        return;

      case DataType::BOOL:
        buffer_->print<bool>(linear);
        return;

      default:
        std::cout << "ERROR: Bad data type!" << std::endl;
        assert(false);
    }
}

} // namespace deeplib
