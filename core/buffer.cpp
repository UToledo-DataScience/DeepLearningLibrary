#include <cassert>
#include <vector>
#include <cstring>
#include "core/buffer.h"
#include "core/utils.h"

namespace deeplib {

Buffer::Buffer() {}

Buffer::Buffer(Buffer* buf) {
    buffer_data_ = nullptr;

    shape_ = buf->getShape();

    total_size_ = buf->getSize();
    total_elements_ = buf->getElements();
    void* buf_ptr = buf->buffer_data_;

    dtype_ = buf->getDataType();

    allocator_ = buf->getAllocator();

    initialize();
    memcpy(buffer_data_, buf_ptr, total_size_);
}

Buffer::Buffer(Buffer* buf, DataType dtype) {
    buffer_data_ = nullptr;

    shape_ = buf->getShape();

    total_elements_ = buf->getElements();
    void* buf_ptr = buf->buffer_data_;

    dtype_ = dtype;

    allocator_ = buf->getAllocator();

    initialize();
}

Buffer::Buffer(std::vector<int> s, Allocator* a) {
    total_size_ = 1;
    for (int i : s)
        total_size_ *= i;

    total_elements_ = total_size_;

    // NOTE: Total_size at this point in the function
    //       does not accurately represent the total amount
    //       allocated for buffer_data.
    //
    //       This will need changed.

    allocator_ = a;

    dtype_ = DataType::FLOAT32;

    buffer_data_ = nullptr;

    shape_ = s;
}

Buffer::Buffer(std::vector<int> values, std::vector<int>& s, Allocator* a) {
    total_size_ = 1;
    for (int i : s)
        total_size_ *= i;

    total_elements_ = total_size_;

    allocator_ = a;

    dtype_ = DataType::INT32;

    int* buf_d = (int*)(allocator_->allocate<int>(total_elements_));
    total_size_ = total_elements_ *  sizeof(int);

    for (int i = 0; i < values.size(); i++)
        buf_d[i] = values[i];

    buffer_data_ = (void*)buf_d;

    shape_ = s;
}

Buffer::~Buffer() {}

void Buffer::initialize() {
    if (buffer_data_ == nullptr) {
        switch (dtype_) {
          case DataType::UINT8:
            buffer_data_ = allocator_->allocate<uint8_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(uint8_t);
            return;

          case DataType::UINT16:
            buffer_data_ = allocator_->allocate<uint16_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(uint16_t);
            return;

          case DataType::UINT32:
            buffer_data_ = allocator_->allocate<uint32_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(uint32_t);
            return;

          case DataType::UINT64:
            buffer_data_ = allocator_->allocate<uint64_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(uint64_t);
            return;

          case DataType::INT8:
            buffer_data_ = allocator_->allocate<int8_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(int8_t);
            return;

          case DataType::INT16:
            buffer_data_ = allocator_->allocate<int16_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(int16_t);
            return;

          case DataType::INT32:
            buffer_data_ = allocator_->allocate<int32_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(int32_t);
            return;

          case DataType::INT64:
            buffer_data_ = allocator_->allocate<int64_t>(total_elements_);
            total_size_ = total_elements_ *  sizeof(int64_t);
            return;

          case DataType::FLOAT32:
            buffer_data_ = allocator_->allocate<float>(total_elements_);
            total_size_ = total_elements_ *  sizeof(float);
            return;

          case DataType::FLOAT64:
            buffer_data_ = allocator_->allocate<double>(total_elements_);
            total_size_ = total_elements_ *  sizeof(double);
            return;

          case DataType::BOOL:
            buffer_data_ = allocator_->allocate<bool>(total_elements_);
            total_size_ = total_elements_ *  sizeof(bool);
            return;

          default:
            std::cout << "ERROR: bad data type, buffer_data not allocated!" << std::endl;
            assert(false);
        }
    }
}

std::vector<int>& Buffer::getShape() {
    return shape_;
}

Allocator* Buffer::getAllocator() {
    return allocator_;
}

DataType Buffer::getDataType() {
    return dtype_;
}

void Buffer::setDataType(DataType new_dtype) {
    dtype_ = new_dtype;
}

uint64_t Buffer::getSize() {
    return total_size_;
}

uint64_t Buffer::getElements() {
    uint64_t total = 1;
    for (int i : shape_)
        total *= i;

    return total;
}

} // namespace deeplib
