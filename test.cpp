#include <iostream>
#include <vector>
#include <chrono>
#include "core/tensor.h"
#include "core/op_functions.h"
#include "core/data_types.h"

using std::cout; using std::endl; using std::vector;
using namespace std::chrono;
using namespace deeplib;

// Operations displayed below are := f(x, y) = (x + ((exp(x * y))^y * x)^1 - y) / x)
// where x == { 2, 2, ... } and y == { 3, 3, ... }
void basicOperations() {
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
    t1 = cast(t1, DataType::FLOAT32);
    t2 = cast(t2, DataType::FLOAT32);
    Tensor tConst({ 1 }, { 1 }, &a);
    tConst = cast(tConst, DataType::FLOAT32);
    Tensor t3 = multiply(t1, t2);

    // TODO: operator overloading

    t3 = exp(t3);
    t3 = multiply(t3, t1); // Replacing variables maintains the graph.
    t3 = power(t3, tConst); // Broadcasting with constants.
    t3 = add(t3, t1); // Reuse of tensors maintains the graph (this is the third use of t1).
    t3 = sub(t3, t2);
    t3 = divide(t3, t1);
    t3 = sqrt(t3);
    t3 = add(t3, t1);

    auto time1 = high_resolution_clock::now();
    t3.operate();
    auto time2 = high_resolution_clock::now();

    cout << "Execution time: "
         << duration_cast<microseconds>(time2 - time1).count() << " microseconds"
         << endl;

    t3.print();
    t3.uproot();

    a.printStats();
}

void matrixOperations() {
    Allocator a;

    vector<int> m1 = { 1, 2, 3,
                       4, 5, 6,
                       7, 8, 9 };

    vector<int> m2 = { 9, 8, 7,
                       6, 5, 4,
                       3, 2, 1 };

    vector<int> shape = { 1, 3, 3 };
    vector<int>& s = shape;

    Tensor t1 = Tensor(m1, s, &a);
    Tensor t2 = Tensor(m2, s, &a);

    Tensor t3 = matmul(t1, t2);

    t3.operate();
    t3.print();
    t3.uproot();

    a.printStats();
}

int main() {
    matrixOperations();
}
