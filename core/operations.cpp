#include <iostream>
#include <cmath>
#include "core/operations.h"

using std::string;

namespace deeplib {

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
template <typename OpDType>
Operation<OpDType>::Operation() {
    parent1_ = nullptr;
    parent2_ = nullptr;
    buffer_ = nullptr;
}

// operation graph node for element-wise multiplication
// only to be used within Tensor<T> objects
template <typename OpDType>
Multiplication<OpDType>::Multiplication(Operation<OpDType>* p1, Operation<OpDType>* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "multiplication";
}

template <typename OpDType>
void Multiplication<OpDType>::setBuffer(Buffer<OpDType>* buf) { this->buffer_ = buf; }

template <typename OpDType>
Buffer<OpDType>* Multiplication<OpDType>::getBuffer() { return this->buffer_; }

template <typename OpDType>
void Multiplication<OpDType>::derive() {
    //OpDType a = this->parent1_->operate() * this->parent2_->derive();
    //OpDType b = this->parent2_->operate() * this->parent1_->derive();

    //return a + b;
}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
template <typename OpDType>
Buffer<OpDType>* Multiplication<OpDType>::operate() {
    this->buffer_->initialize();

    Buffer<OpDType>* p1;
    Buffer<OpDType>* p2;

    p1 = this->parent1_->operate();
    p2 = this->parent2_->operate();

    for (uint64_t i = 0; i < p1->getSize(); i++)
        this->buffer_->setIndex(i, p1->getIndex(i) * p2->getIndex(i));

    return this->buffer_;
}

template <typename OpDType>
Power<OpDType>::Power(Operation<OpDType>* p1, Operation<OpDType>* p2) {
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->type_ = "power";
}

template <typename OpDType>
void Power<OpDType>::setBuffer(Buffer<OpDType>* buf) { this->buffer_ = buf; }

template <typename OpDType>
Buffer<OpDType>* Power<OpDType>::getBuffer() { return this->buffer_; }

template <typename OpDType>
void Power<OpDType>::derive() {
    //OpDType a = this->parent1_->operate() * this->parent2_->derive();
    //OpDType b = this->parent2_->operate() * this->parent1_->derive();

    //return a + b;
}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
template <typename OpDType>
Buffer<OpDType>* Power<OpDType>::operate() {
    this->buffer_->initialize();

    Buffer<OpDType>* p1;
    Buffer<OpDType>* p2;

    p1 = this->parent1_->operate();
    p2 = this->parent2_->operate();

    // NOTE: using std::pow here is temporary and will have to change
    //       it's only here right now for foundational purposes
    for (uint64_t i = 0; i < p1->getSize(); i++)
        this->buffer_->setIndex(i, std::pow(p1->getIndex(i), p2->getIndex(0)));

    return this->buffer_;
}

template <typename OpDType>
Constant<OpDType>::Constant(Buffer<OpDType>* buf) {
    this->buffer_ = buf;
    this->type_ = "constant";
}

template <typename OpDType>
void Constant<OpDType>::setBuffer(Buffer<OpDType>* buf) { this->buffer_ = buf; }

template <typename OpDType>
Buffer<OpDType>* Constant<OpDType>::getBuffer() { return this->buffer_; }

template <typename OpDType>
void Constant<OpDType>::derive() {
    return;
}

template <typename OpDType>
Buffer<OpDType>* Constant<OpDType>::operate() {
    return this->buffer_;
}

} // namespace deeplib
