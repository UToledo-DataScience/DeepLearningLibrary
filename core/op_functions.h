#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

template <typename T>
Tensor<T> mult(Tensor<T>* t1, Tensor<T>* t2) {
    if (t1 == t2)
        return pow(t1, new Tensor<T>({ 2, 2 }, { 2 }, t1->getAllocator()));

    return Tensor<T>(t1, t2, new operations::Multiplication<T>(t1->getOperation(), t2->getOperation()));
}

template <typename T>
Tensor<T> pow(Tensor<T>* t1, Tensor<T>* t2) {
    return Tensor<T>(t1, t2, new operations::Power<T>(t1->getOperation(), t2->getOperation()));
}

} // namespace deeplib
#endif
