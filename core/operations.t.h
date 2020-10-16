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
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(b1->getIndex<OpDType>(0) / b2->getIndex<OpDType>(i)));
    }
    else if (b2->getElements() == 1) {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(b1->getIndex<OpDType>(i) / b2->getIndex<OpDType>(0)));
    }
    else {
        for (uint64_t i = 0; i < b1->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(b1->getIndex<OpDType>(i) / b2->getIndex<OpDType>(i)));
    }
}

// Naive matrix multiplication algorithm.
template <typename OpDType>
void MatrixMultiplication::compute(Buffer* b1, Buffer* b2) {
    std::vector<int>& shape1 = b1->getShape();
    std::vector<int>& shape2 = b2->getShape();

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
                    v1 = b1->getIndex<OpDType>(start_indices[0]+r*in_cols1+v);
                    v2 = b2->getIndex<OpDType>(start_indices[1]+v*in_cols2+c);
                    vecdot += v1 * v2;
                }

                this->buffer_->setIndex<OpDType>(start_indices[2]+r*out_cols+c, vecdot);
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

    int (&strides)[2] = this->strides_;

    bool padded;
    int padding_offset_y[2], padding_offset_x[2];
    if (!this->padding_.compare("same")) {
        padded = true;

        padding_offset_y[0] = std::ceil(static_cast<float>(kernel_shape[0] - 1) / 2);
        padding_offset_y[1] = std::floor(static_cast<float>(kernel_shape[0] - 1) / 2);

        padding_offset_x[0] = std::ceil(static_cast<float>(kernel_shape[1] - 1) / 2);
        std::cout << padding_offset_x[0] << std::endl;
        padding_offset_x[1] = std::floor(static_cast<float>(kernel_shape[1] - 1) / 2);
        std::cout << padding_offset_x[1] << std::endl;
    }
    else {
        padded = false;

        padding_offset_y[0] = 0;
        padding_offset_y[1] = 0;

        padding_offset_x[0] = 0;
        padding_offset_x[1] = 0;
    }

    int matrix_count = 1;
    int matrix_sizes[3] = { b1_shape[b1_shape.size()-2]*b1_shape[b1_shape.size()-1],
                            kernel_shape[b1_shape.size()-2]*kernel_shape[kernel_shape.size()-1],
                            b1_shape[b1_shape.size()-2]*kernel_shape[kernel_shape.size()-1] };

    for (int i = 0; i < b1_shape.size()-2; i++)
        matrix_count *= b1_shape[i];

    for (int i = 0; i < matrix_count; i++) {
        int start_indices[3] = { matrix_sizes[0]*i,
                                 matrix_sizes[1]*i,
                                 matrix_sizes[2]*i };

        // oy and ox refer to the position of the top left corner of the kernel as it slides across the image.
        for (int oy = 0-padding_offset_y[0]; oy < image_shape[0]-kernel_shape[0]+1+padding_offset_y[1]; oy+=strides[0]) {
            for (int ox = 0-padding_offset_x[0]; ox < image_shape[1]-kernel_shape[1]+1+padding_offset_x[1]; ox+=strides[1]) {
                OpDType local_sum = 0;
                // The actual computation between the kernel and the image.
                // Indices prefixed with k refer to the kernel, i refer to the image.
                for (int ky = kernel_shape[0]-1, iy = oy;
                     ky > -1, iy < oy+kernel_shape[0];
                     ky--, iy++) {
                    for (int kx = kernel_shape[1]-1, ix = ox;
                         kx > -1, ix < ox+kernel_shape[1];
                         kx--, ix++) {
                        int image_index = start_indices[0]+iy*image_shape[1]+ix;
                        int kernel_index = start_indices[1]+ky*kernel_shape[1]+kx;

                        bool padding_condition_y = (iy < 0 || iy >= image_shape[0]);
                        bool padding_condition_x = (ix < 0 || ix >= image_shape[1]);
                        bool padding_condition = padding_condition_y || padding_condition_x;

                        if (padding_condition)
                            continue;

                        OpDType ik1 = b1->getIndex<OpDType>(image_index);
                        OpDType ik2 = b2->getIndex<OpDType>(kernel_index);

                        local_sum += ik1 * ik2;
                    }
                }

                int out_y, out_x;
                if (padded && strides[0] == 1 && strides[1] == 1) {
                    out_y = oy + padding_offset_y[0];
                    out_x = ox + padding_offset_x[0];
                }
                else {
                    out_y = std::floor(oy / strides[0]);
                    out_x = std::floor(ox / strides[1]);
                }

                this->buffer_->setIndex<OpDType>(start_indices[2]+out_y*output_shape[1]+out_x, local_sum);
            }
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
        this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(std::exp(buf->getIndex<OpDType>(i))));
}

// NOTE: OpDType refers to this->buffer_->dtype.
//       Another switch statement is done in
template <typename OpDType>
void Cast::compute(Buffer* buf) {
    switch (buf->getDataType()) {
      case DataType::UINT8:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<uint16_t>(i)));
        return;

      case DataType::UINT16:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<uint16_t>(i)));
        return;

      case DataType::UINT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<uint32_t>(i)));
        return;

      case DataType::UINT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<uint64_t>(i)));
        return;

      case DataType::INT8:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<int8_t>(i)));
        return;

      case DataType::INT16:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<int16_t>(i)));
        return;

      case DataType::INT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<int32_t>(i)));
        return;

      case DataType::INT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<int64_t>(i)));
        return;

      case DataType::FLOAT32:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<float>(i)));
        return;

      case DataType::FLOAT64:
        for (uint64_t i = 0; i < buf->getElements(); i++)
            this->buffer_->setIndex<OpDType>(i, static_cast<OpDType>(buf->getIndex<double>(i)));
        return;
    }
}

} // namespace deeplib
