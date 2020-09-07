#ifndef OPERATIONS
#define OPERATIONS
#include <iostream>
#include <cmath>
#include "core/buffer.h"

using std::string;

namespace deeplib {

class Buffer;
class Allocator;

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
class Operation {
    friend class Allocator;

  protected:
    string name_;
    string type_;

    Operation* parent1_;
    Operation* parent2_;

    Buffer* buffer_;

  public:
    Operation();

    Operation(Operation* p1, Operation* p2);
    Operation(Buffer* buf);

    virtual void setBuffer(Buffer* buf) = 0;
    virtual Buffer* getBuffer() = 0;

    virtual void derive() = 0;

    virtual Buffer* operate() = 0;

    string getType();
};

// operation graph node for element-wise multiplication
// only to be used within Tensor<T> objects
class Multiplication : public Operation {
  public:
    Multiplication(Operation* p1, Operation* p2);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    //template <typename OpDType>
    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};

class Power : public Operation {
  public:
    Power(Operation* p1, Operation* p2);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    // NOTE: broadcasting not yet supported
    //
    // element-wise multiplication - no shape change
    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};

class Constant : public Operation {
  public:
    Constant(Buffer* buf);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();
};

} // namespace deeplib

#include "core/operations.t.h"
#endif
