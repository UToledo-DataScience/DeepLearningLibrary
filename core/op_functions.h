#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// NOTE: It is assumed that the allocators of t1 and t2 are the same.
//       this may change in the future.

Tensor power(Tensor* t1, Tensor* t2) {
    assert(t1->getDataType() == t2->getDataType());
    
    return Tensor(t1, t2,
        t1->getAllocator()->newOperation(
            new Power(t1->getOperation(), t2->getOperation())));
}

Tensor mult(Tensor* t1, Tensor* t2) {
    assert(t1->getDataType() == t2->getDataType());

    if (t1 == t2) {
        Tensor pt2({ 2, 2 }, { 2 }, t1->getAllocator());
        return power(t1, &pt2);
    }

    return Tensor(t1, t2,
        t1->getAllocator()->newOperation(
            new Multiplication(t1->getOperation(), t2->getOperation())));
}

} // namespace deeplib
#endif
