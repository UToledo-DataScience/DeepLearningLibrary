#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/utils.h"

namespace deeplib {

template <typename BDType>
Buffer<BDType>::Buffer() {}

// copy constructor
template <typename BDType>
Buffer<BDType>::Buffer(Buffer<BDType>* buf) {
    shape = buf->getShape();

    total_size = buf->getSize();
    BDType* buf_ptr = buf->getBufferData();

    allocator = buf->getAllocator();

    buffer_data = allocator->allocate(total_size);
    for (uint64_t i = 0; i < total_size; i++)
        buffer_data[i] = buf_ptr[i];
}

template <typename BDType>
Buffer<BDType>::Buffer(std::vector<int> s, Allocator<BDType>* a) {
    total_size = 1;
    for (int i : s)
        total_size *= i;

    allocator = a;

    buffer_data = nullptr;

    shape = s;
}

// if the user provides values to initialize from
template <typename BDType>
Buffer<BDType>::Buffer(std::vector<BDType> values, std::vector<int>& s, Allocator<BDType>* a) {
    total_size = 1;
    for (int i : s)
        total_size *= i;

    allocator = a;

    buffer_data = allocator->allocate(total_size);
    for (int i = 0; i < values.size(); i++)
        buffer_data[i] = values[i];

    shape = s;
}

// TODO: REFERENCE COUNTS
template <typename BDType>
Buffer<BDType>::~Buffer() {
    //delete placeholder;
    //delete operation;
}

template <typename BDType>
void Buffer<BDType>::initialize() {
    if (buffer_data == nullptr)
        buffer_data = allocator->allocate(total_size);
}

// returns the value at the given index
template <typename BDType>
BDType Buffer<BDType>::getIndex(uint64_t index) {
    assert(buffer_data != nullptr);
    assert(index < total_size);
    return buffer_data[index];
}

// sets the value at the given index
template <typename BDType>
void Buffer<BDType>::setIndex(uint64_t index, BDType value) {
    assert(buffer_data != nullptr);
    assert(index < total_size);

    buffer_data[index] = value;
}

template <typename BDType>
BDType* Buffer<BDType>::getBufferData() { return buffer_data; }

template <typename BDType>
std::vector<int>& Buffer<BDType>::getShape() { return shape; }

template <typename BDType>
Allocator<BDType>* Buffer<BDType>::getAllocator() { return allocator; }

template <typename BDType>
uint64_t Buffer<BDType>::getSize() {
    uint64_t total = 1;
    for (int i : shape)
        total *= i;

    return total;
}

// naive print function
// obviously ill-suited for higher dimensions
// and sizes
//
// only deals with the last two dimensions
template <typename BDType>
void Buffer<BDType>::print() {
    int r = *(shape.end()-1);
    int c = *(shape.end()-2);
    
    if (shape.size() == 1) {
        c = shape[0];
        r = 1;
    }

    for (int i = 0; i < r; i++) {
        std::cout << "[ ";
        for (int j = 0; j < c; j++)
            std::cout << buffer_data[i*c+j] << " ";

        std::cout << "]" << std::endl;
    }
}

} // namespace deeplib
