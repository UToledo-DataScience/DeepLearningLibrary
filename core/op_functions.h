#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// NOTE: It is assumed that the allocators of t1 and t2 are the same.
//       this may change in the future.

Tensor add(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Addition(t1.getOperation(), t2.getOperation())));
}

Tensor sub(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Subtraction(t1.getOperation(), t2.getOperation())));
}

Tensor power(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());
    
    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Power(t1.getOperation(), t2.getOperation())));
}

Tensor multiply(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    if (&t1 == &t2) {
       return power(t1, t2);
    }

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Multiplication(t1.getOperation(), t2.getOperation())));
}

Tensor divide(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Division(t1.getOperation(), t2.getOperation())));
}

Tensor sqrt(Tensor& t) {
    DataType dtype = t.getDataType();
    if (dtype < DataType::FLOAT32) {
        std::cout << "ERROR: Data type of operation sqrt must be floating point!" << std::endl;
        assert(false);
    }
    else {
        return Tensor(t, 
            t.getAllocator()->newOperation(
                new SquareRoot(t.getOperation())));
    }
}

Tensor cast(Tensor& t, DataType new_dtype) {
    return Tensor(t,
        t.getAllocator()->newOperation(
            new Cast(t.getOperation())), new_dtype);
}

Tensor exp(Tensor& t) {
    return Tensor(t, 
        t.getAllocator()->newOperation(
            new Exponential(t.getOperation())));
}

} // namespace deeplib
#endif
