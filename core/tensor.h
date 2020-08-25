#ifndef TENSOR
#define TENSOR
#include <iostream>
#include <cassert>
#include <vector>
#include "core/placeholder.h"
#include "core/operations.h"
#include "core/utils.h"

template <typename TensorDType>
class Tensor {
    Placeholder<TensorDType>* placeholder;
    std::vector<int> shape;

    Operation<TensorDType>* operation;

    public:
        // fresh tensor
        Tensor(std::vector<int> s, Allocator<TensorDType>* allocator) {
            int total_size = 1;
            for (int i : s) {
                shape.push_back(i);
                total_size *= i;
            }

            placeholder = new Placeholder<TensorDType>(allocator, total_size);
            operation = new Constant<TensorDType>(placeholder);
        }

        // tensor from binary operation
        Tensor(Tensor<TensorDType>* t1, Tensor<TensorDType>* t2, std::string op) {
            assert(compare(t1->getShape(), t2->getShape()));

            // TODO: accomodate the other operations, as well as different shapes
            //       i.e. make better
            
            for (int i : t1->getShape())
                shape.push_back(i);

            placeholder = t1->getPlaceholder();
            operation = new Multiplication(t1->getOperation(), t2->getOperation());
        }

        // TODO: REFERENCE COUNTS
        ~Tensor() {
            //delete placeholder;
            //delete operation;
        }

        TensorDType operate() { return operation->operate(); }

        Placeholder<TensorDType>* getPlaceholder() { return placeholder; }

        std::vector<int>& getShape() { return shape; }

        Operation<TensorDType>* getOperation() { return operation; }

        void print() {
            placeholder->print(shape);
        }
};
#endif
