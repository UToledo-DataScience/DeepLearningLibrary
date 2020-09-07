#include <iostream>
#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/operations.h"
#include "core/utils.h"

namespace deeplib {

Allocator::Allocator(): total_allocations_(0),
                                 total_deallocations_(0),
                                 bytes_allocated_(0),
                                 bytes_deallocated_(0),
                                 bytes_currently_allocated_(0)
    {}

Allocator::~Allocator() {
    uproot();
}

Operation* Allocator::newOperation(Operation* new_op) {
    operations_.push_back(new_op);

    bytes_allocated_ += sizeof(Operation);
    bytes_currently_allocated_ += sizeof(Operation);
    total_allocations_++;

    return new_op;
}

Buffer* Allocator::newBuffer(Buffer* new_buf) {
    buffers_.push_back(new_buf);

    bytes_allocated_ += sizeof(Buffer);
    bytes_currently_allocated_ += sizeof(Buffer);
    total_allocations_++;

    return new_buf;
}

void Allocator::freeBuffer(Buffer* buf) {
    if (buf->buffer_data_ != nullptr) {
        free(buf->buffer_data_);
        buf->buffer_data_ = nullptr;

        delete buf;

        uint64_t dealloc_size = buf->total_size_ + sizeof(Buffer);

        bytes_deallocated_ += dealloc_size;
        bytes_currently_allocated_ -= dealloc_size;
        total_deallocations_++;
    }
}

void Allocator::uproot() {
    for (auto buf : buffers_)
        freeBuffer(buf);

    for (auto oper : operations_) {
        delete oper;
        bytes_deallocated_ += sizeof(Operation);
        bytes_currently_allocated_ -= sizeof(Operation);
        total_deallocations_++;
    }

    buffers_.clear();
    operations_.clear();
}

void Allocator::uprootOperation(Operation* op, int& index) {
    Operation* p1 = op->parent1_;
    Operation* p2 = op->parent2_;

    // The actual deallocation of the operation
    // and it's related buffer.

    int i = in<Buffer>(op->buffer_, buffers_);
    if (i > -1) {
        freeBuffer(op->buffer_);
        buffers_.erase(buffers_.begin() + i);
    }

    delete op;
    operations_.erase(operations_.begin() + index);
    bytes_deallocated_ += sizeof(Operation);
    bytes_currently_allocated_ -= sizeof(Operation);
    total_deallocations_++;

    // Recursively travel up the tree and deallocate.

    index = in<Operation>(p1, operations_);
    if (p1 != nullptr && index > -1)
        uprootOperation(p1, index);

    index = in<Operation>(p2, operations_);
    if (p2 != nullptr && index > -1)
        uprootOperation(p2, index);
}

void Allocator::uprootOperation(Operation* op) {
    Buffer* buf = op->buffer_;

    int i = in<Buffer>(buf, buffers_);
    if (i > -1) {
        freeBuffer(buf);
        buffers_.erase(buffers_.begin() + i);
    }

    i = in<Operation>(op, operations_);
    if (i > -1)
        uprootOperation(op, i);
}

void Allocator::printStats() {
    std::cout << "total_allocations_: " << total_allocations_ << std::endl
              << "total_deallocations_: " << total_deallocations_ << std::endl
              << "bytes_allocated_: " << bytes_allocated_ << std::endl
              << "bytes_deallocated_: " << bytes_deallocated_ << std::endl
              << "bytes_currently_allocated_: " << bytes_currently_allocated_ << std::endl;
}

} // namespace deeplib
