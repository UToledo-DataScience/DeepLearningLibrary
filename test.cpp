#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include "core/tensor.h"
#include "core/graph.h"
#include "core/op_functions.h"
#include "core/data_types.h"

using std::cout; using std::endl; using std::vector;
using std::map;
using namespace std::chrono;
using namespace deeplib;

void eval(Tensor& t) {
    auto time1 = high_resolution_clock::now();
    t.operate();
    auto time2 = high_resolution_clock::now();

    cout << "Execution time: "
         << duration_cast<microseconds>(time2 - time1).count() << " microseconds"
         << endl;

    t.print();
    t.uproot();
}

// Operations displayed below are := f(x, y) = (x + ((exp(x * y))^y * x)^1 - y) / x)
// where x == { 2, 2, ... } and y == { 3, 3, ... }
void basicOperations() {
    Allocator a;

    int size = 5;

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
    Tensor t4 = power(t3, tConst); // Broadcasting with constants.
    t4 = add(t4, t1); // Reuse of tensors maintains the graph (this is the third use of t1).
    t4 = sub(t4, t2);
    t4 = divide(t4, t1);
    Tensor t5 = sqrt(t1);
    Tensor t6 = exp(t2);
    Tensor t7 = add(t5, t6);
    Tensor t8 = divide(t4, t7);

    t8.print();
    t8.uproot();

    a.printStats();
}

void graphTest() {
    Allocator a;

    int size = 5;

    vector<int> v11;
    vector<int> v22;

    for (int i = 0; i < size; i++) {
        v11.push_back(2);
        v22.push_back(3);
    }

    vector<int> s = { size };

    Tensor t1(v11, s, &a);
    Tensor t2(v22, s, &a);
    Tensor ct1 = cast(t1, DataType::FLOAT32);
    Tensor ct2 = cast(t2, DataType::FLOAT32);
    Tensor tConst({ 1 }, { 1 }, &a);
    tConst = cast(tConst, DataType::FLOAT32);
    Tensor t3 = multiply(ct1, ct2);

    // TODO: operator overloading

    t3 = exp(t3);
    t3 = multiply(t3, ct1); // Replacing variables maintains the graph.
    Tensor t4 = power(t3, tConst); // Broadcasting with constants.
    t4 = add(t4, ct1); // Reuse of tensors maintains the graph (this is the third use of t1).
    t4 = sub(t4, ct2);
    t4 = divide(t4, ct1);
    Tensor t5 = sqrt(ct1);
    Tensor t6 = exp(ct2);
    Tensor t7 = add(t5, t6);
    Tensor t8 = divide(t4, t7);

    vector<Tensor> leaves = { t4, t5 };

    Graph graph(t8, leaves, &a);
    graph.traceGraph();

    vector<int> v3, v4;

    for (int i = 0; i < size; i++) {
        v3.push_back((i+1)*(i+1));
        v4.push_back((i+1)*(i+1));
    }

    map<string, Tensor> filler;

    std::vector<Tensor> results = graph.graphComputation(filler);

    results[0].print();

    graph.uproot();


    cout << "Graph uproot:" << endl;
    a.printStats();

    cout << "-------------------------------" << endl;
    a.uproot();
    a.printStats();

}

void matrixMultiplication() {
    Allocator a;

    vector<int> m1 = { 1, 1, 1,
                       1, 1, 1,
                       1, 1, 1 };

    vector<int> m2 = { 2, 2, 2,
                       2, 2, 2,
                       2, 2, 2 };

    vector<int> shape = { 3, 3 };

    Tensor t1(m1, shape, &a);
    Tensor t2(m2, shape, &a);

    Tensor t3 = matmul(t1, t2);

    t3.operate();
    t3.print();
    t3.uproot();

    a.printStats();
}

void convolution() {
    Allocator a;

    vector<int> m1;
    for (int i = 0; i < 100; i++)
        m1.push_back(i+1);

    vector<int> m2 = { 1, 1, 2,
                       2, 2, 2,
                       1, 1, 1 };

    vector<int> shape = { 10, 10 };
    vector<int> shape2 = { 3, 3 };

    int strides[2] = { 3, 3 };

    Tensor t1 = Tensor(m1, shape, &a);
    Tensor t2 = Tensor(m2, shape2, &a);

    Tensor t3 = conv2d(t1, t2, "valid", strides);

    t3.operate();
    t3.print();
    t3.uproot();

    a.printStats();
}

int main() {
    graphTest();
}
