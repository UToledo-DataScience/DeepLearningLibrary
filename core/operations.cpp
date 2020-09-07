#include <iostream>
#include <cmath>
#include "core/operations.h"

using std::string;

namespace deeplib {

//-----------------------------------\\
// class Operation;                  \\
//-----------------------------------\\

// Abstract operation graph node class.
//
// Each Operation has two key functions:
//   - Operation.derive()
//   - Operation.operate()
//
// derive() is the function corresponding to the
// derivative of that specific operation.
// e.g. Multiplication.derive == d/dx(x * x) == x*1 + 1*x == product rule
//
// operate() is a recursive used to actually enact the arithmetic
// defined by the operation.
// e.g. Multiply.operate() == x * y,
//      where x == Operation.parent1_ and y == Operation.parent2_
//
// It should be noted that Constant.operate() retrieves the actual value,
// acting as the end condition to the recursive function
// as newly created Tensor<T> objects have a Constant as their operation. // TODO: review this last line
//
// This is never to be used directly.
Operation::Operation() {
    parent1_ = nullptr;
    parent2_ = nullptr;
    buffer_ = nullptr;
}

//-----------------------------------\\
// class Multiplication;             \\
//-----------------------------------\\

// operation graph node for element-wise multiplication
// only to be used within Tensor<T> objects
Multiplication::Multiplication(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "multiplication";
}

void Multiplication::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Multiplication::getBuffer() { return this->buffer_; }

void Multiplication::derive() {
    //OpDType a = this->parent1_->operate() * this->parent2_->derive();
    //OpDType b = this->parent2_->operate() * this->parent1_->derive();

    //return a + b;
}

// NOTE: broadcasting not yet supported
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Multiplication::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    // there HAS to be a better way of dealing with data types
    // please let me know
    switch (dtype) {
      case DataType::UINT8:
        this->compute<uint8_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT16:
        this->compute<uint16_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT32:
        this->compute<uint32_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT64:
        this->compute<uint64_t>(b1, b2);
        return this->buffer_;

      case DataType::INT8:
        this->compute<int8_t>(b1, b2);
        return this->buffer_;

      case DataType::INT16:
        this->compute<int16_t>(b1, b2);
        return this->buffer_;

      case DataType::INT32:
        this->compute<int32_t>(b1, b2);
        return this->buffer_;

      case DataType::INT64:
        this->compute<int64_t>(b1, b2);
        return this->buffer_;
            
      case DataType::FLOAT32:
        this->compute<float>(b1, b2);
        return this->buffer_;

      case DataType::FLOAT64:
        this->compute<double>(b1, b2);
        return this->buffer_;

      case DataType::BOOL:
        this->compute<bool>(b1, b2);
        return this->buffer_;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

//-----------------------------------\\
// class Power;                      \\
//-----------------------------------\\

Power::Power(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "power";
}

void Power::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Power::getBuffer() { return this->buffer_; }

void Power::derive() {
    //OpDType a = this->parent1_->operate() * this->parent2_->derive();
    //OpDType b = this->parent2_->operate() * this->parent1_->derive();

    //return a + b;
}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Power::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    switch (dtype) {
      case DataType::UINT8:
        this->compute<uint8_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT16:
        this->compute<uint16_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT32:
        this->compute<uint32_t>(b1, b2);
        return this->buffer_;

      case DataType::UINT64:
        this->compute<uint64_t>(b1, b2);
        return this->buffer_;

      case DataType::INT8:
        this->compute<int8_t>(b1, b2);
        return this->buffer_;

      case DataType::INT16:
        this->compute<int16_t>(b1, b2);
        return this->buffer_;

      case DataType::INT32:
        this->compute<int32_t>(b1, b2);
        return this->buffer_;

      case DataType::INT64:
        this->compute<int64_t>(b1, b2);
        return this->buffer_;
            
      case DataType::FLOAT32:
        this->compute<float>(b1, b2);
        return this->buffer_;

      case DataType::FLOAT64:
        this->compute<double>(b1, b2);
        return this->buffer_;

      case DataType::BOOL:
        this->compute<bool>(b1, b2);
        return this->buffer_;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

//-----------------------------------\\
// class Constant;                   \\
//-----------------------------------\\

Constant::Constant(Buffer* buf) {
    this->buffer_ = buf;
    this->type_ = "constant";
}

void Constant::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Constant::getBuffer() { return this->buffer_; }

void Constant::derive() {
    return;
}

Buffer* Constant::operate() {
    return this->buffer_;
}

} // namespace deeplib
