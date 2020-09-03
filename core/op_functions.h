#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// NOTE: It is assumed that the allocators of t1 and t2 are the same.
//       this may change in the future.

template <typename T>
Tensor<T> mult(Tensor<T>* t1, Tensor<T>* t2) {
    if (t1 == t2) {
        Tensor<T> pt2({ 2, 2 }, { 2 }, t1->getAllocator());
        return pow(t1, &pt2);
    }

    return Tensor<T>(t1, t2,
        t1->getAllocator()->newOperation(
            new Multiplication<T>(t1->getOperation(), t2->getOperation())));
}

template <typename T>
Tensor<T> pow(Tensor<T>* t1, Tensor<T>* t2) {
    return Tensor<T>(t1, t2,
        t1->getAllocator()->newOperation(
            new Power<T>(t1->getOperation(), t2->getOperation())));
}

} // namespace deeplib
#endif
