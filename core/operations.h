#ifndef OPERATIONS
#define OPERATIONS
#include <iostream>
#include <cmath>
#include "core/buffer.h"

using std::string;

namespace deeplib {

template <typename T>
class Buffer;

template <typename T>
class Allocator;

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
    friend class Allocator<OpDType>;

  protected:
    string name_;
    string type_;

    Operation<OpDType>* parent1_;
    Operation<OpDType>* parent2_;

    Buffer<OpDType>* buffer_;

  public:
    Operation();

    Operation(Operation<OpDType>* p1, Operation<OpDType>* p2);
    Operation(Buffer<OpDType>* buf);

    virtual void setBuffer(Buffer<OpDType>* buf) = 0;
    virtual Buffer<OpDType>* getBuffer() = 0;

    virtual void derive() = 0;
    virtual Buffer<OpDType>* operate() = 0;

    string getType();
};

// operation graph node for element-wise multiplication
// only to be used within Tensor<T> objects
template <typename OpDType>
class Multiplication : public Operation<OpDType> {
  public:
    Multiplication(Operation<OpDType>* p1, Operation<OpDType>* p2);

    void setBuffer(Buffer<OpDType>* buf);
    Buffer<OpDType>* getBuffer();

    void derive();

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    Buffer<OpDType>* operate();
};

template <typename OpDType>
class Power : public Operation<OpDType> {
  public:
    Power(Operation<OpDType>* p1, Operation<OpDType>* p2);

    void setBuffer(Buffer<OpDType>* buf);
    Buffer<OpDType>* getBuffer();

    void derive();

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    Buffer<OpDType>* operate();
};

template <typename OpDType>
class Constant : public Operation<OpDType> {
  public:
    Constant(Buffer<OpDType>* buf);

    void setBuffer(Buffer<OpDType>* buf);
    Buffer<OpDType>* getBuffer();

    void derive();

    Buffer<OpDType>* operate();
};

} // namespace deeplib

#include "core/operations.cpp"
#endif
