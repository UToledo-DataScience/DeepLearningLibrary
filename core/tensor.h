#ifndef TENSOR
#define TENSOR
#include <iostream>
#include <vector>
#include "core/placeholder.h"

template <class T>
class Tensor {
    Placeholder<T> placeholder;
    std::vector<int> shape;

    public:
        Tensor(std::vector<int> s, Allocator<T>* allocator) {
            int total_size = 1;
            for (int i : s) {
                shape.push_back(i);
                total_size *= i;
            }

            placeholder.initialize(allocator, total_size);
        }

        void print() {
            placeholder.print(shape);
        }
};
#endif
