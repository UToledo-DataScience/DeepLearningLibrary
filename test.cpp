#include <iostream>
#include <vector>
#include "core/placeholder.h"
#include "core/tensor.h"

using namespace std;

int main() {
    Allocator<int> a;

    std::vector<int> shape = { 10, 10 };
    Tensor<int> t(shape, &a);

    t.print();
    a.printStats();
}
