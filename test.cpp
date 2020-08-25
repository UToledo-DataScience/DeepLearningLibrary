#include <iostream>
#include <vector>
#include "core/placeholder.h"
#include "core/tensor.h"
#include "core/operations.h"

using namespace std;

// f(x) = x^3
// df/dx = 3x^2

int main() {
    Allocator<int> a;

    // default value of 2
    Tensor<int> t1({ 1 }, &a);
    Tensor<int> t2({ 1 }, &a);
    Tensor<int> t3({ 1 }, &a);

    Tensor<int> t4(&t1, &t2, "multiplication");
    Tensor<int> t5(&t4, &t3, "multiplication");

    // f(2) = 8
    cout << t5.operate() << endl;

    Tensor<int> t6(&t5, &t5, "multiplication");

    // f(2) * f(2) = 64
    cout << t6.operate() << endl;

    a.printStats();
}
