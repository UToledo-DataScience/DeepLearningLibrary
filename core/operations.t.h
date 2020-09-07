namespace deeplib {

template <typename OpDType>
void Multiplication::compute(Buffer* b1, Buffer* b2) {
    for (uint64_t i = 0; i < b1->getElements(); i++)
        this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) * b2->getIndex<OpDType>(i));
}

template <typename OpDType>
void Power::compute(Buffer* b1, Buffer* b2) {
    // NOTE: using std::pow here is temporary and will have to change
    //       it's only here right now for foundational purposes
    for (uint64_t i = 0; i < b1->getElements(); i++)
        this->buffer_->setIndex<OpDType>(i, std::pow(b1->getIndex<OpDType>(i), b2->getIndex<OpDType>(0)));
}

} // namespace deeplib
