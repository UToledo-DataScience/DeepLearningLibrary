#ifndef TENSOR_BACKEND
#define TENSOR_BACKEND
#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/data_types.h"

namespace deeplib {

class Allocator;

class Buffer {
    friend class Allocator;

    void* buffer_data;

    DataType dtype;

    Allocator* allocator;

    std::vector<int> shape;

    uint64_t total_size; // This refers to the initial number of bytes that has been allocated.
                         // It should be noted that the original shape is what has been allocated.
                         // The same allocation will be used for sizes <= the original size.
                         // If the size should be >, then a new tensor is allocated.

    uint64_t total_elements; // Total number of elements managed by this buffer. This number
                             // will change as the shape of buffer_data changes.

  public:
    Buffer();

    // copy constructor
    Buffer(Buffer* buf);

    Buffer(std::vector<int> s, Allocator* a);

    // if the user provides values to initialize from
    Buffer(std::vector<int> values, std::vector<int>& s, Allocator* a);

    // TODO: REFERENCE COUNTS
    ~Buffer();

    void initialize();

    // returns the value at the given index
    template <typename BDType>
    BDType getIndex(uint64_t index);

    // sets the value at the given index
    template <typename BDType>
    void setIndex(uint64_t index, BDType value);

    template <typename BDType>
    BDType* getBufferDataAsTemplate();

    void* getBufferData();

    DataType getDataType();

    std::vector<int>& getShape();

    Allocator* getAllocator();

    uint64_t getSize();
    uint64_t getElements();

    // naive print function
    // obviously ill-suited for higher dimensions
    // and sizes
    //
    // only deals with the last two dimensions
    template <typename BDType>
    void print();
};

} // namespace deeplib

#include "core/buffer.t.h"
#endif // #ifndef TENSOR_BACKEND
