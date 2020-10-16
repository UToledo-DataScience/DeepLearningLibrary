#include <iostream>
#include <cmath>
#include "core/operations.h"

using std::string;

namespace deeplib {

// Feed an Operation type (not Operation itself though) in here along with the buffers
// to avoid unnecessary repeating of the switch statements.
// 
// A better method than the previous in avoiding code bloat (I hope).
template <class Op>
void compTemplateChoice(Op* op, Buffer* b1, Buffer* b2, DataType dtype) {
    switch (dtype) {
      case DataType::UINT8:
        op->template compute<uint8_t>(b1, b2);
        return;

      case DataType::UINT16:
        op->template compute<uint16_t>(b1, b2);
        return;

      case DataType::UINT32:
        op->template compute<uint32_t>(b1, b2);
        return;

      case DataType::UINT64:
        op->template compute<uint64_t>(b1, b2);
        return;

      case DataType::INT8:
        op->template compute<int8_t>(b1, b2);
        return;

      case DataType::INT16:
        op->template compute<int16_t>(b1, b2);
        return;

      case DataType::INT32:
        op->template compute<int32_t>(b1, b2);
        return;

      case DataType::INT64:
        op->template compute<int64_t>(b1, b2);
        return;
            
      case DataType::FLOAT32:
        op->template compute<float>(b1, b2);
        return;

      case DataType::FLOAT64:
        op->template compute<double>(b1, b2);
        return;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

// Overloaded for unary operations.
template <class Op>
void compTemplateChoice(Op* op, Buffer* b1, DataType dtype) {
    switch (dtype) {
      case DataType::UINT8:
        op->template compute<uint8_t>(b1);
        return;

      case DataType::UINT16:
        op->template compute<uint16_t>(b1);
        return;

      case DataType::UINT32:
        op->template compute<uint32_t>(b1);
        return;

      case DataType::UINT64:
        op->template compute<uint64_t>(b1);
        return;

      case DataType::INT8:
        op->template compute<int8_t>(b1);
        return;

      case DataType::INT16:
        op->template compute<int16_t>(b1);
        return;

      case DataType::INT32:
        op->template compute<int32_t>(b1);
        return;

      case DataType::INT64:
        op->template compute<int64_t>(b1);
        return;
            
      case DataType::FLOAT32:
        op->template compute<float>(b1);
        return;

      case DataType::FLOAT64:
        op->template compute<double>(b1);
        return;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

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
// class Addition;                   \\
//-----------------------------------\\

// Operation graph node for element-wise multiplication.
Addition::Addition(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "addition";
}

void Addition::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Addition::getBuffer() { return this->buffer_; }

void Addition::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Addition::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Addition>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Subtraction;                \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Subtraction::Subtraction(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "subtraction";
}

void Subtraction::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Subtraction::getBuffer() { return this->buffer_; }

void Subtraction::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Subtraction::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Subtraction>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Multiplication;             \\
//-----------------------------------\\

// Operation graph node for element-wise multiplication.
Multiplication::Multiplication(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "multiplication";
}

void Multiplication::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Multiplication::getBuffer() { return this->buffer_; }

void Multiplication::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Multiplication::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Multiplication>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Division;                   \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Division::Division(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "division";
}

void Division::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Division::getBuffer() { return this->buffer_; }

void Division::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Division::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Division>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class MatrixMultiplication;       \\
//-----------------------------------\\

// Operation graph node for element-wise division.
MatrixMultiplication::MatrixMultiplication(Operation* p1, Operation* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "matrix_multiplication";
}

void MatrixMultiplication::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* MatrixMultiplication::getBuffer() { return this->buffer_; }

void MatrixMultiplication::derive() {}

Buffer* MatrixMultiplication::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<MatrixMultiplication>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Convolution2D;              \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Convolution2D::Convolution2D(Operation* p1, Operation* p2, std::string padding, int (&strides)[2]) {
    this->padding_ = padding;
    this->strides_[0] = strides[0];
    this->strides_[1] = strides[1];
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "convolution2d";
}

void Convolution2D::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Convolution2D::getBuffer() { return this->buffer_; }

void Convolution2D::derive() {}

Buffer* Convolution2D::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Convolution2D>(this, b1, b2, dtype);
    return this->buffer_;
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

void Power::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Power::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Power>(this, b1, b2, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Cast;                       \\
//-----------------------------------\\

Cast::Cast(Operation* p1) {
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->type_ = "cast";
}

void Cast::setBuffer(Buffer* buf) {
    this->buffer_ = buf;
}

Buffer* Cast::getBuffer() {
    return this->buffer_;
}

void Cast::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Cast::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    DataType dtype = this->buffer_->getDataType();

    compTemplateChoice<Cast>(this, buf, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class SquareRoot;                 \\
//-----------------------------------\\

SquareRoot::SquareRoot(Operation* p1, bool promotion) {
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->type_ = "square_root";
    this->promotion = promotion;
}

void SquareRoot::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* SquareRoot::getBuffer() { return this->buffer_; }

void SquareRoot::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* SquareRoot::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    DataType dtype;
    if (this->promotion)
        dtype = DataType::FLOAT32;
    else
        dtype = this->buffer_->getDataType();

    compTemplateChoice<SquareRoot>(this, buf, dtype);
    return this->buffer_;
}

//-----------------------------------\\
// class Exponential;                \\
//-----------------------------------\\

Exponential::Exponential(Operation* p1) {
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->type_ = "exponential";
}

void Exponential::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Exponential::getBuffer() { return this->buffer_; }

void Exponential::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Exponential::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    DataType dtype = buf->getDataType();

    compTemplateChoice<Exponential>(this, buf, dtype);
    return this->buffer_;
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
