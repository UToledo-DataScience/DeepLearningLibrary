#include <iostream>
#include <vector>
#include <chrono>
#include "core/tensor.h"
#include "core/op_functions.h"

using namespace std;
using namespace std::chrono;
using namespace deeplib;

// Operations displayed below are := f(x, y) = ((x^2 * y)^y)^2
// where x == { 2, 2, ... } and y == { 3, 3, ... }

int main() {
    Allocator a;

    int size = 10;

    vector<int> v11;
    vector<int> v22;

    for (int i = 0; i < size; i++) {
        v11.push_back(2);
        v22.push_back(3);
    }

    vector<int> s = { size };

    Tensor t1(v11, s, &a);
    Tensor t2(v22, s, &a);
    Tensor t3 = mult(&t1, &t2);

    t3 = power(&t3, &t2);
    t3 = mult(&t3, &t1); // replacing variables maintains the graph
    t3 = mult(&t3, &t3); // using binary operators for unary operation also maintains graph
                         // e.g. here it is t3 * t3 == t3^2

    auto time1 = high_resolution_clock::now();
    t3.operate();
    auto time2 = high_resolution_clock::now();

    cout << "Execution time: "
         << duration_cast<nanoseconds>(time2 - time1).count() << " nanoseconds"
         << endl;

    t3.print();
    t3.uproot();

    a.printStats();
}
