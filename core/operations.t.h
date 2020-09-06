#include "core/operations.h"

namespace deeplib {

template <typename OpDType>
Buffer* Operation::operate() {
    return nullptr;
}

// NOTE: broadcasting not yet supported
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
template <typename OpDType>
Buffer* Multiplication::operate() {
    this->buffer_->initialize();

    Buffer* p1; 
    Buffer* p2; 

    p1 = this->parent1_->operate<OpDType>();
    p2 = this->parent2_->operate<OpDType>();

    for (uint64_t i = 0; i < p1->getSize(); i++)
        this->buffer_->setIndex<OpDType>(i, p1->getIndex<OpDType>(i) * p2->getIndex<OpDType>(i));

    return this->buffer_;
}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
template <typename OpDType>
Buffer* Power::operate() {
    this->buffer_->initialize();

    Buffer* p1;
    Buffer* p2;

    p1 = this->parent1_->operate<OpDType>();
    p2 = this->parent2_->operate<OpDType>();

    // NOTE: using std::pow here is temporary and will have to change
    //       it's only here right now for foundational purposes
    for (uint64_t i = 0; i < p1->getSize(); i++)
        this->buffer_->setIndex<OpDType>(i, std::pow(p1->getIndex<OpDType>(i), p2->getIndex<OpDType>(0)));

    return this->buffer_;
}

template <typename OpDType>
Buffer* Constant::operate() {
    return this->buffer_;
}

} // namespace deeplib
