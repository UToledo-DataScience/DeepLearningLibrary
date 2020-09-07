#include <cassert>
#include <vector>
#include <cstring>
#include "core/buffer.h"
#include "core/allocator.h"
#include "core/utils.h"

namespace deeplib {

Buffer::Buffer() {}

// copy constructor
Buffer::Buffer(Buffer* buf) {
    buffer_data = nullptr;

    shape = buf->getShape();

    total_size = buf->getSize();
    total_elements = buf->getElements();
    void* buf_ptr = buf->getBufferData();

    dtype = buf->getDataType();

    allocator = buf->getAllocator();

    initialize();
    memcpy(buffer_data, buf_ptr, total_size);
}

Buffer::Buffer(std::vector<int> s, Allocator* a) {
    total_size = 1;
    for (int i : s)
        total_size *= i;

    total_elements = total_size;

    // NOTE: Total_size at this point in the function
    //       does not accurately represent the total amount
    //       allocated for buffer_data.
    //
    //       This will need changed.

    allocator = a;

    dtype = DataType::FLOAT32;

    buffer_data = nullptr;

    shape = s;
}

// If the user provides values from which to initialize.
//
// NOTE: this is not permanent. A better method will have to come about
//       for initializing tensors from values.
Buffer::Buffer(std::vector<int> values, std::vector<int>& s, Allocator* a) {
    total_size = 1;
    for (int i : s)
        total_size *= i;

    total_elements = total_size;

    allocator = a;

    dtype = DataType::INT32;

    int* buf_d = (int*)(allocator->allocate<int>(total_elements));
    total_size = total_elements *  sizeof(int);

    //#pragma omp parallel for
    for (int i = 0; i < values.size(); i++)
        buf_d[i] = values[i];

    buffer_data = (void*)buf_d;

    shape = s;
}

// TODO: REFERENCE COUNTS
Buffer::~Buffer() {}

void Buffer::initialize() {
    if (buffer_data == nullptr) {
        switch (dtype) {
          case DataType::UINT8:
            buffer_data = allocator->allocate<uint8_t>(total_elements);
            total_size = total_elements *  sizeof(uint8_t);
            return;

          case DataType::UINT16:
            buffer_data = allocator->allocate<uint16_t>(total_elements);
            total_size = total_elements *  sizeof(uint16_t);
            return;

          case DataType::UINT32:
            buffer_data = allocator->allocate<uint32_t>(total_elements);
            total_size = total_elements *  sizeof(uint32_t);
            return;

          case DataType::UINT64:
            buffer_data = allocator->allocate<uint64_t>(total_elements);
            total_size = total_elements *  sizeof(uint64_t);
            return;

          case DataType::INT8:
            buffer_data = allocator->allocate<int8_t>(total_elements);
            total_size = total_elements *  sizeof(int8_t);
            return;

          case DataType::INT16:
            buffer_data = allocator->allocate<int16_t>(total_elements);
            total_size = total_elements *  sizeof(int16_t);
            return;

          case DataType::INT32:
            buffer_data = allocator->allocate<int32_t>(total_elements);
            total_size = total_elements *  sizeof(int32_t);
            return;

          case DataType::INT64:
            buffer_data = allocator->allocate<int64_t>(total_elements);
            total_size = total_elements *  sizeof(int64_t);
            return;

          case DataType::FLOAT32:
            buffer_data = allocator->allocate<float>(total_elements);
            total_size = total_elements *  sizeof(float);
            return;

          case DataType::FLOAT64:
            buffer_data = allocator->allocate<double>(total_elements);
            total_size = total_elements *  sizeof(double);
            return;

          case DataType::BOOL:
            buffer_data = allocator->allocate<bool>(total_elements);
            total_size = total_elements *  sizeof(bool);
            return;

          default:
            std::cout << "ERROR: bad data type, buffer_data not allocated!" << std::endl;
            assert(false);
        }
    }
}

void* Buffer::getBufferData() { return buffer_data; }

std::vector<int>& Buffer::getShape() { return shape; }

Allocator* Buffer::getAllocator() { return allocator; }

DataType Buffer::getDataType() { return dtype; }

uint64_t Buffer::getSize() { return total_size; }

uint64_t Buffer::getElements() {
    uint64_t total = 1;
    for (int i : shape)
        total *= i;

    return total;
}

} // namespace deeplib
