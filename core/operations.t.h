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

// Naive matrix multiplication algorithm.
template <typename OpDType>
void MatrixMultiplication::compute(Buffer* b1, Buffer* b2) {
    std::vector<int>& shape1 = b1->getShape();
    std::vector<int>& shape2 = b2->getShape();

    if (shape1.size() > 2) {
        int matrix_count = 1;
        int matrix_sizes[3] = { shape1[shape1.size()-2]*shape1[shape1.size()-1],
                                shape2[shape2.size()-2]*shape2[shape2.size()-1],
                                shape1[shape1.size()-2]*shape2[shape2.size()-1] };

        for (int i = 0; i < shape1.size()-2; i++)
            matrix_count *= shape1[i];

        for (int i = 0; i < matrix_count; i++) {
            int start_indices[3] = { matrix_sizes[0]*i,
                                     matrix_sizes[1]*i,
                                     matrix_sizes[2]*i };

            int in_cols1 = shape1[shape1.size()-1];
            int in_cols2 = shape2[shape2.size()-2];

            // Output rows and columns.
            int out_rows = shape1[shape1.size()-2];
            int out_cols = shape2[shape2.size()-1];

            int vec_length = shape1[shape1.size()-1];

            for (int r = 0; r < out_rows; r++) {
                for (int c = 0; c < out_cols; c++) {
                    OpDType vecdot = 0;
                    for (int v = 0; v < vec_length; v++) {
                        OpDType v1, v2;
                        v1 = (OpDType)(b1->getIndex<OpDType>(start_indices[0]+r*in_cols1+v));
                        v2 = (OpDType)(b2->getIndex<OpDType>(start_indices[1]+v*in_cols2+c));
                        vecdot += v1 * v2;
                    }

                    this->buffer_->setIndex<OpDType>(start_indices[2]+r*out_cols+c, vecdot);
                }
            }
        }
    }
    else {
        int in_cols1 = shape1[shape1.size()-1];
        int in_cols2 = shape2[shape2.size()-2];

        // Output rows and columns.
        int out_rows = shape1[shape1.size()-2];
        int out_cols = shape2[shape2.size()-1];

        int vec_length = shape1[shape1.size()-1];

        for (int r = 0; r < out_rows; r++) {
            for (int c = 0; c < out_cols; c++) {
                OpDType vecdot = 0;
                for (int v = 0; v < vec_length; v++) {
                    OpDType v1, v2;
                    v1 = (OpDType)(b1->getIndex<OpDType>(r*in_cols1+v));
                    v2 = (OpDType)(b2->getIndex<OpDType>(v*in_cols2+c));
                    vecdot += v1 * v2;
                }

                this->buffer_->setIndex<OpDType>(r*out_cols+c, vecdot);
            }
        }
    }
}

// NOTE: kernel shape (i.e. b2->getShape()) will always be ND for ConvolutionND
template <typename OpDType>
void Convolution2D::compute(Buffer* b1, Buffer* b2) {
    std::vector<int>& b1_shape = b1->getShape();
    std::vector<int> image_shape;
    for (int s = b1_shape.size()-2; s < b1_shape.size(); s++)
        image_shape.push_back(b1_shape[s]);

    std::vector<int>& kernel_shape = b2->getShape();
    std::vector<int>& output_shape = this->buffer_->getShape();

    // NOTE: Fix for shapes > 2 !

    // (y, x) == (0, 0) will refer to the top left of the image
    int kernel_offset_y = std::ceil(kernel_shape[0] / 2) - 1;
    int kernel_offset_x = std::ceil(kernel_shape[1] / 2) - 1;

    // Perform the iterations from the coordinate view of the output image.
    for (int oy = kernel_offset_y; oy < image_shape[0]-kernel_offset_y; oy++) {
        for (int ox = kernel_offset_x; ox < image_shape[0]-kernel_offset_x; ox++) {
            OpDType local_sum = 0;
            // Indices prefixed with k refer to the kernel, i refer to the image.
            for (int ky = kernel_shape[0]-1, iy = oy-kernel_offset_y;
                 ky > -1, iy < oy+kernel_offset_y+1;
                 ky--, iy++) {
                for (int kx = kernel_shape[1]-1, ix = ox-kernel_offset_x;
                     kx > -1, ix < ox+kernel_offset_x+1;
                     kx--, ix++) {
                    // NOTE: This needs changed for shapes > 2 !
                    local_sum += b1->getIndex<OpDType>(iy*image_shape[1]+ix) * b2->getIndex<OpDType>(ky*kernel_shape[1]+kx);
                }
            }

            this->buffer_->setIndex<OpDType>(oy*output_shape[1]+ox, local_sum);
        }
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
