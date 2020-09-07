namespace deeplib {

// naive print function
// obviously ill-suited for higher dimensions
// and sizes
//
// only deals with the last two dimensions
template <typename BDType>
void Buffer::print() {
    int r = *(shape_.end()-1);
    int c = *(shape_.end()-2);

    if (shape_.size() == 1) {
        c = shape_[0];
        r = 1;
    }

    BDType* data = (BDType*)buffer_data_;

    for (int i = 0; i < r; i++) {
        std::cout << "[ ";
        for (int j = 0; j < c; j++)
            std::cout << data[i*c+j] << " ";

        std::cout << "]" << std::endl;
    }
}

// returns the value at the given index
template <typename BDType>
BDType Buffer::getIndex(uint64_t index) {
    assert(buffer_data_ != nullptr);
    assert(index < total_elements_);
    return ((BDType*)buffer_data_)[index];
}

// sets the value at the given index
template <typename BDType>
void Buffer::setIndex(uint64_t index, BDType value) {
    assert(buffer_data_ != nullptr);
    assert(index < total_size_);

    BDType* temp = (BDType*)buffer_data_;
    temp[index] = value;
    buffer_data_ = (void*)buffer_data_;
}

template <typename BDType>
BDType* Buffer::getBufferDataAsTemplate() {
    return static_cast<BDType*>(buffer_data_);
}

} // namespace deeplib
