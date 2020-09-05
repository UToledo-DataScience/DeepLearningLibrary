#include <iostream>
#include <vector>
#include "core/tensor.h"
#include "core/op_functions.h"

using namespace std;
using namespace deeplib;

// Operations displayed below are := f(x, y) = (x^2 * y)^2
// where x == { 2, 2 } and y == { 3, 2 }

int main() {
    Allocator<int> a;

    vector<int> v11 = { 2, 2 };
    vector<int> v22 = { 3, 2 };

    vector<int> s = { 2 };

    Tensor<int> t1(v11, s, &a);
    Tensor<int> t2(v22, s, &a);

    Tensor<int> t3 = mult(&t1, &t2);
    t3 = mult(&t3, &t1); // replacing variables maintains the graph

    t3 = mult(&t3, &t3); // using binary operators for unary operation also maintains graph
                         // e.g. here it is t3 * t3 == t3^2
    t3.operate();
    t3.print();

    a.uproot(&t3);

    a.printStats();
}
