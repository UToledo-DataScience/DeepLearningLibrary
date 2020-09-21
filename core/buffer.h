#ifndef TENSOR_BACKEND
#define TENSOR_BACKEND
#include <cassert>
#include <vector>
#include "core/allocator.h"
#include "core/data_types.h"

namespace deeplib {

class Allocator;

// Buffer class intended hold allocated data for tensors in the graph.
// Basically a wrapper for the dynamically allocated data pointer,
// which here is void* buffer_data.
//
//----CREATION RULES----
// This class has special creation and use rules regarding memory
// allocation in order to try and keep things memory efficient.
// New buffers will only be allocated from:
//   - A binary tensor operation in which one of the parents has
//     more than 0 children.
//   - A tensor created from predetermined values.
//
// In a binary operation in which a new buffer is not created,
// the buffer will be shared and overwritten once the
// corresponding operation's .operate() is called.
//
// TODO: refine the creation rules, it feels too ad hoc.
class Buffer {
    friend class Allocator;
    friend class Cast; // For ease in changing buffer data types.

    void* buffer_data_;

    DataType dtype_;

    Allocator* allocator_;

    std::vector<int> shape_;
 
    // The initial number of bytes that has been allocated.
    // It should be noted that the original shape is what has been allocated.
    // The same allocation will be used for sizes <= the original size.
    // If the size should be >, then a new tensor is allocated.
    uint64_t total_size_;

    // Total number of elements managed by this buffer. This number
    // will change as the shape of buffer_data changes.
    uint64_t total_elements_;

  public:
    Buffer();

    // Custom copy constructor.
    Buffer(Buffer* buf);

    // Custom copy constructor for implicit cast-conversions.
    // NOTE: This constructor does not copy buf->buffer_data_, it only 
    //       allocates this->buffer_data_ based on buf->getElements()
    Buffer(Buffer* buf, DataType new_dtype);

    // Uninitialized buffer.
    // TODO: expand on this.
    Buffer(std::vector<int> s, Allocator* a);

    // If the user provides values to initialize from.
    // Allocated buffer_data is initialized with the given
    // set of values.
    // TODO: make this method of constructing better. Maybe
    //       incorporate Eigen tensors?
    Buffer(std::vector<int> values, std::vector<int>& s, Allocator* a);

    // TODO: Is there a cleaner way of destruction?
    ~Buffer();

    // If buffer_data is nullptr i.e. unallocated, then
    // this allocates buffer_data. Otherwise, it does nothing.
    void initialize();

    // Returns the value at the given index.
    template <typename BDType>
    BDType getIndex(uint64_t index);

    // Sets the value at the given index.
    template <typename BDType>
    void setIndex(uint64_t index, BDType value);

    // Self-explanatory getters.

    template <typename BDType>
    BDType* getBufferDataAsTemplate();

    DataType getDataType();

    void setDataType(DataType new_dtype);

    std::vector<int>& getShape();

    Allocator* getAllocator();

    uint64_t getSize();
    uint64_t getElements();

    // Naive print function obviously ill-suited
    // for higher dimensions and sizes.
    //
    // Currently only deals with the last two dimensions. 
    template <typename BDType>
    void print();
};

} // namespace deeplib

#include "core/buffer.t.h"
#endif // #ifndef TENSOR_BACKEND
