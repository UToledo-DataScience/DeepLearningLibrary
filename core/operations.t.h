#include <cmath>

namespace deeplib {

// NOTE: Only broadcasting with constants is supported right now.
// NOTE: How long will STL math functions be used?

template <typename OpDType>
void Addition::compute(Buffer* b1, Buffer* b2) {
    if (b1->getElements() == 1) {
        for (uint64_t i = 0; i < b2->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(0) + b2->getIndex<OpDType>(i));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) + b2->getIndex<OpDType>(0));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) + b2->getIndex<OpDType>(i));
    }
}

template <typename OpDType>
void Subtraction::compute(Buffer* b1, Buffer* b2) {
    if (b1->getElements() == 1) {
        for (uint64_t i = 0; i < b2->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(0) - b2->getIndex<OpDType>(i));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) - b2->getIndex<OpDType>(0));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) - b2->getIndex<OpDType>(i));
    }
}

template <typename OpDType>
void Multiplication::compute(Buffer* b1, Buffer* b2) {
    if (b1->getElements() == 1) {
        for (uint64_t i = 0; i < b2->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(0) * b2->getIndex<OpDType>(i));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) * b2->getIndex<OpDType>(0));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, b1->getIndex<OpDType>(i) * b2->getIndex<OpDType>(i));
    }
}

template <typename OpDType>
void Division::compute(Buffer* b1, Buffer* b2) {
    if (b1->getElements() == 1) {
        for (uint64_t i = 0; i < b2->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(b1->getIndex<OpDType>(0) / b2->getIndex<OpDType>(i)));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(b1->getIndex<OpDType>(i) / b2->getIndex<OpDType>(0)));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(b1->getIndex<OpDType>(i) / b2->getIndex<OpDType>(i)));
    }
}

template <typename OpDType>
void Power::compute(Buffer* b1, Buffer* b2) {
    if (b1->getElements() == 1) {
        for (uint64_t i = 0; i < b2->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, std::pow(b1->getIndex<OpDType>(0), b2->getIndex<OpDType>(i)));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, std::pow(b1->getIndex<OpDType>(i), b2->getIndex<OpDType>(0)));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, std::pow(b1->getIndex<OpDType>(i), b2->getIndex<OpDType>(i)));
    }
}

template <typename OpDType>
void SquareRoot::compute(Buffer* buf) {
    switch (buf->getDataType()) {
      case DataType::FLOAT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (std::sqrt(buf->getIndex<float>(i))));
        return;

      case DataType::FLOAT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (std::sqrt(buf->getIndex<double>(i))));
        return;

      default:
        std::cout << "ERROR: Data type must be floating point in sqrt!" << std::endl;
        assert(false);
    }
}

template <typename OpDType>
void Exponential::compute(Buffer* buf) {
    for (uint64_t i = 0; i < buf->getElements(); i++)
        this->buffer_->setIndex<OpDType>(i, (OpDType)(std::exp(buf->getIndex<OpDType>(i))));
}

// NOTE: OpDType refers to this->buffer_->dtype.
//       Another switch statement is done in
template <typename OpDType>
void Cast::compute(Buffer* buf) {
    switch (buf->getDataType()) {
      case DataType::UINT8:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<uint16_t>(i)));
        return;

      case DataType::UINT16:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<uint16_t>(i)));
        return;

      case DataType::UINT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<uint32_t>(i)));
        return;

      case DataType::UINT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<uint64_t>(i)));
        return;

      case DataType::INT8:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<int8_t>(i)));
        return;

      case DataType::INT16:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<int16_t>(i)));
        return;

      case DataType::INT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<int32_t>(i)));
        return;

      case DataType::INT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<int64_t>(i)));
        return;

      case DataType::FLOAT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<float>(i)));
        return;

      case DataType::FLOAT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, (OpDType)(buf->getIndex<double>(i)));
        return;
    }
}

} // namespace deeplib
