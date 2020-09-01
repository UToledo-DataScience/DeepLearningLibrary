#ifndef OPERATIONS
#define OPERATIONS
#include <iostream>
#include <cmath>
#include "core/buffer.h"

using std::string;
using deeplib::Buffer;

namespace operations {

/*const string MULT;
const string DIV;
const string ADD;
const string SUB;
const string EXP;
const string LOG;
const string POW;*/

// TODO: how closely tied should tensors and operations be?

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
class Operation {
  protected:
    string name_;
    string type_;

    Operation<OpDType>* parent1_;
    Operation<OpDType>* parent2_;

    std::shared_ptr<Buffer<OpDType>> buffer_;

  public:
    Operation() {
        parent1_ = nullptr;
        parent2_ = nullptr;
        buffer_ = nullptr;
    }

    Operation(Operation<OpDType>* p1, Operation<OpDType>* p2);
    Operation(std::shared_ptr<Buffer<OpDType>> buf);

    virtual void setBuffer(std::shared_ptr<Buffer<OpDType>> buf) = 0;
    virtual void derive() = 0;
    virtual std::shared_ptr<Buffer<OpDType>> operate() = 0;

    string getType();
};

// operation graph node for element-wise multiplication
// only to be used within Tensor<T> objects
template <typename OpDType>
class Multiplication : public Operation<OpDType> {
  //private:
  //  OpDType multiply(Buffer<OpDType>* p1, Buffer<OpDType>* p2,
  //                   std::vector<OpDType>& shape1, std::vector<OpDType>& shape2) {

  public:
    Multiplication(Operation<OpDType>* p1, Operation<OpDType>* p2) {
        this->parent1_ = p1;
        this->parent2_ = p2;
        this->type_ = "multiplication";
    }

    void setBuffer(std::shared_ptr<Buffer<OpDType>> buf) { this->buffer_ = buf; }

    void derive() {
        //OpDType a = this->parent1_->operate() * this->parent2_->derive();
        //OpDType b = this->parent2_->operate() * this->parent1_->derive();

        //return a + b;
    }

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    std::shared_ptr<Buffer<OpDType>> operate() {
        this->buffer_->initialize();

        std::shared_ptr<Buffer<OpDType>> p1, p2;

        p1 = this->parent1_->operate();
        p2 = this->parent2_->operate();

        for (uint64_t i = 0; i < p1->getSize(); i++)
            this->buffer_->setIndex(i, p1->getIndex(i) * p2->getIndex(i));

        return this->buffer_;
    }
};

template <typename OpDType>
class Power : public Operation<OpDType> {
  public:
    Power(Operation<OpDType>* p1, Operation<OpDType>* p2) {
        this->parent1_ = p1;
        this->parent2_ = p2;
        this->type_ = "power";
    }

    void setBuffer(std::shared_ptr<Buffer<OpDType>> buf) { this->buffer_ = buf; }

    void derive() {
        //OpDType a = this->parent1_->operate() * this->parent2_->derive();
        //OpDType b = this->parent2_->operate() * this->parent1_->derive();

        //return a + b;
    }

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    std::shared_ptr<Buffer<OpDType>> operate() {
        this->buffer_->initialize();

        std::shared_ptr<Buffer<OpDType>> p1, p2;

        p1 = this->parent1_->operate();
        p2 = this->parent2_->operate();

        // NOTE: using std::pow here is temporary and will have to change
        //       it's only here right now for foundational purposes
        for (uint64_t i = 0; i < p1->getSize(); i++)
            this->buffer_->setIndex(i, std::pow(p1->getIndex(i), p2->getIndex(0)));

        return this->buffer_;
    }
};

template <typename OpDType>
class Constant : public Operation<OpDType> {
  public:
    Constant(std::shared_ptr<Buffer<OpDType>> buf) {
        this->buffer_ = buf;
        this->type_ = "constant";
    }

    void setBuffer(std::shared_ptr<Buffer<OpDType>> buf) { this->buffer_ = buf; }

    void derive() {
        return;
    }

    std::shared_ptr<Buffer<OpDType>> operate() {
        return this->buffer_;
    }
};

} // namespace operations
#endif
