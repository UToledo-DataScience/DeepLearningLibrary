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

class Addition : public Operation {
  public:
    Addition(Operation* p1, Operation* p2);

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

class Subtraction : public Operation {
  public:
    Subtraction(Operation* p1, Operation* p2);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};


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

class Division : public Operation {
  public:
    Division(Operation* p1, Operation* p2);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};

class MatrixMultiplication : public Operation {
  public:
    MatrixMultiplication(Operation* p1, Operation* p2);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};

class Convolution2D : public Operation {
    int strides_[2];
    std::string padding_;

  public:
    Convolution2D(Operation* p1, Operation* p2, std::string padding, int (&strides)[2]);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

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

    // Element-wise raising to a power - no shape change
    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* b1, Buffer* b2);
};

class Cast : public Operation {
  public:
    Cast(Operation* buf);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* buf);
};

class SquareRoot : public Operation {
    // Tensors going through this operation can be promoted
    // to float32 if they're signed and not a floating point tensor
    bool promotion;

  public:
    SquareRoot(Operation* p, bool promotion=false);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* buf);
};

class Exponential : public Operation {
  public:
    Exponential(Operation* p);

    void setBuffer(Buffer* buf);
    Buffer* getBuffer();

    void derive();

    // Element-wise exp() function - no shape change.
    Buffer* operate();

    template <typename OpDType>
    void compute(Buffer* buf);
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
