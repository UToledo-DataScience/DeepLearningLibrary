#include <cmath>
#include "core/computation_kernels.h"

namespace deeplib {

// NOTE: Only broadcasting with constants is supported right now.
// NOTE: How long will STL math functions be used?

template <typename OpDType>
void Addition::compute(Buffer* b1, Buffer* b2) {
    buffer_addition<OpDType>(b1, b2, this->buffer_);
}

template <typename OpDType>
void Subtraction::compute(Buffer* b1, Buffer* b2) {
    buffer_subtraction<OpDType>(b1, b2, this->buffer_);
}

template <typename OpDType>
void Multiplication::compute(Buffer* b1, Buffer* b2) {
    buffer_multiplication<OpDType>(b1, b2, this->buffer_);
}

/*template <typename OpDType>
void Multiplication::gradientCompute(Buffer* b1, Buffer* b2, Buffer* grad1, Buffer* grad2, Buffer* compute_buffer) {
    buffer_multiplication<OpDType>(b1, grad2, compute_buffer);
    buffer_multiplication<OpDType>(b2, grad1, this->gradient_output_buffer);

    buffer_addition<OpDType>(compute_buffer, this->gradient_output_buffer, this->gradient_output_buffer);
}*/

template <typename OpDType>
void Division::compute(Buffer* b1, Buffer* b2) {
    buffer_division<OpDType>(b1, b2, this->buffer_);
}

// Naive matrix multiplication algorithm.
template <typename OpDType>
void MatrixMultiplication::compute(Buffer* b1, Buffer* b2) {
    buffer_matrix_multiplication<OpDType>(b1, b2, this->buffer_);
}

// NOTE: kernel shape (i.e. b2->getShape()) will always be ND for ConvolutionND
template <typename OpDType>
void Convolution2D::compute(Buffer* b1, Buffer* b2) {
    buffer_convolution2d<OpDType>(b1, b2, this->buffer_, this->strides_, this->padding_);
}

template <typename OpDType>
void Power::compute(Buffer* b1, Buffer* b2) {
    buffer_power<OpDType>(b1, b2, this->buffer_);
}

template <typename OpDType>
void SquareRoot::compute(Buffer* buf) {
    buffer_square_root<OpDType>(buf, this->buffer_);
}

template <typename OpDType>
void Exponential::compute(Buffer* buf) {
    buffer_exponential<OpDType>(buf, this->buffer_);
}

// NOTE: OpDType refers to this->buffer_->dtype.
//       Another switch statement is done in
template <typename OpDType>
void Cast::compute(Buffer* buf) {
    buffer_cast<OpDType>(buf, this->buffer_);
}

} // namespace deeplib
