#ifndef TENSOR_BACKEND
#define TENSOR_BACKEND
#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/utils.h"

namespace deeplib {

template <typename T>
class Allocator;

template <typename BDType>
class Buffer {
    friend class Allocator<BDType>;

    BDType* buffer_data;

    Allocator<BDType>* allocator;

    std::vector<int> shape;

    uint64_t total_size; // this refers to the initial amount that has been allocated
                         // it should be noted that the original shape is what has been allocated
                         // the same allocation will be used for sizes <= the original size
                         // should the size be >, then a new tensor is allocated

  public:
    Buffer();

    // copy constructor
    Buffer(Buffer<BDType>* buf);

    Buffer(std::vector<int> s, Allocator<BDType>* a);

    // if the user provides values to initialize from
    Buffer(std::vector<BDType> values, std::vector<int>& s, Allocator<BDType>* a);

    // TODO: REFERENCE COUNTS
    ~Buffer();

    void initialize();

    // returns the value at the given index
    BDType getIndex(uint64_t index);

    // sets the value at the given index
    void setIndex(uint64_t index, BDType value);

    BDType* getBufferData();

    std::vector<int>& getShape();

    Allocator<BDType>* getAllocator();

    uint64_t getSize();

    // naive print function
    // obviously ill-suited for higher dimensions
    // and sizes
    //
    // only deals with the last two dimensions
    void print();
};

} // namespace deeplib

#include "core/buffer.cpp"
#endif // #ifndef TENSOR_BACKEND
