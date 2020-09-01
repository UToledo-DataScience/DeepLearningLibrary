#ifndef TENSOR_BACKEND
#define TENSOR_BACKEND
#include <cassert>
#include <vector>
#include "core/placeholder.h"
#include "core/utils.h"

namespace deeplib {

template <typename BDType>
class Buffer {
    BDType* buffer_data;

    Allocator<BDType>* allocator;

    std::vector<int> shape;

    uint64_t total_size; // this refers to the initial amount that has been allocated
                         // it should be noted that the original shape is what has been allocated
                         // the same allocation will be used for sizes <= the original size
                         // should the size be >, then a new tensor is allocated

  public:
    Buffer() {}

    // copy constructor
    Buffer(std::shared_ptr<Buffer<BDType>> buf) {
        shape = buf->getShape();

        total_size = buf->getSize();
        BDType* buf_ptr = buf->getBufferData();

        allocator = buf->getAllocator();

        buffer_data = allocator->allocate(total_size);
        for (uint64_t i = 0; i < total_size; i++)
            buffer_data[i] = buf_ptr[i];
    }

    Buffer(std::vector<int> s, Allocator<BDType>* a) {
        total_size = 1;
        for (int i : s)
            total_size *= i;

        allocator = a;

        buffer_data = nullptr;

        shape = s;
    }

    // if the user provides values to initialize from
    Buffer(std::vector<BDType> values, std::vector<int>& s, Allocator<BDType>* a) {
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
    ~Buffer() {
        //delete placeholder;
        //delete operation;
    }

    void initialize() {
        if (buffer_data == nullptr)
            buffer_data = allocator->allocate(total_size);
    }

            // returns the value at the given index
    BDType getIndex(uint64_t index) {
        assert(buffer_data != nullptr);
        assert(index < total_size);
        return buffer_data[index];
    }

    // sets the value at the given index
    void setIndex(uint64_t index, BDType value) {
        assert(buffer_data != nullptr);
        assert(index < total_size);

        buffer_data[index] = value;
    }

    BDType* getBufferData() { return buffer_data; }

    std::vector<int>& getShape() { return shape; }

    Allocator<BDType>* getAllocator() { return allocator; }

    uint64_t getSize() {
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
    void print() {
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
};

} // namespace deeplib
#endif // #ifndef TENSOR_BACKEND
