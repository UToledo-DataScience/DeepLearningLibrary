#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include <string>
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// NOTE: It is assumed that the allocators of t1 and t2 are the same.
//       this may change in the future.

Tensor add(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Addition(t1.getOperation(), t2.getOperation())), dynamic);
}

Tensor sub(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Subtraction(t1.getOperation(), t2.getOperation())), dynamic);
}

Tensor power(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());
    
    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Power(t1.getOperation(), t2.getOperation())), dynamic);
}

Tensor multiply(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());

    if (&t1 == &t2) {
       return power(t1, t2);
    }

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Multiplication(t1.getOperation(), t2.getOperation())), dynamic);
}

Tensor divide(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Division(t1.getOperation(), t2.getOperation())), dynamic);
}

// Inputs must be at least 2D. Inputs of higher rank
// will only be considered for their last two dimensions.
//
// Shape is assumed to be in format [..., rows, columns]
Tensor matmul(Tensor& t1, Tensor& t2, bool dynamic=true) {
    assert(t1.getDataType() == t2.getDataType());
    std::vector<int>& shape1 = t1.getShape();
    std::vector<int>& shape2 = t2.getShape();

    std::vector<int> new_shape;

    // Shape requirements.
    assert(shape1.size() >= 2 && shape2.size() >= 2);
    for (int i = 0; i < shape1.size()-2; i++) {
        assert(shape1[i] == shape2[i]);
        new_shape.push_back(shape1[i]);
    }

    assert(shape1[shape1.size()-2] == shape2[shape2.size()-1]);

    new_shape.push_back(shape1[shape1.size()-2]);
    new_shape.push_back(shape2[shape2.size()-1]);

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new MatrixMultiplication(t1.getOperation(), t2.getOperation())), new_shape, dynamic);
}

// TODO: dilation_rate
Tensor conv2d(Tensor& image, Tensor& kernel, std::string padding, int (&strides)[2], bool dynamic=true) {
    assert(image.getDataType() == kernel.getDataType());

    assert(strides[0] >= 0 && strides[1] >= 0);

    std::string padding_values[2] = { "same", "valid" };

    std::vector<int>& image_shape = image.getShape();
    std::vector<int>& kernel_shape = kernel.getShape();

    std::vector<int> new_shape;

    int kernel_offset_y = std::ceil(static_cast<float>(kernel_shape[0]) / 2) - 1;
    int kernel_offset_x = std::ceil(static_cast<float>(kernel_shape[1]) / 2) - 1;

    assert(image_shape.size() >= 2 && kernel_shape.size() == 2);
    for (int i = 0; i < image_shape.size()-2; i++)
        new_shape.push_back(image_shape[i]);

    // NOTE: strides override padding
    if (!padding.compare(padding_values[0]) && strides[0] == 1 && strides[1] == 1) {
        new_shape.push_back(image_shape.rbegin()[1]);
        new_shape.push_back(image_shape.back());
    }
    else if (!padding.compare(padding_values[1]) || strides[0] > 1 || strides[1] > 1) {
        new_shape.push_back(std::floor((image_shape.rbegin()[1] - kernel_shape[0])/strides[0]) + 1);
        new_shape.push_back(std::floor((image_shape.back() - kernel_shape[1])/strides[1]) + 1);
    }
    else {
        // Please find a better way of handling this.
        std::cout << "ERROR: Invalid padding value. Must be either \"same\" or \"valid\"" << std::endl;
        assert(false);
    }

    return Tensor(image, kernel,
        image.getAllocator()->newOperation(
            new Convolution2D(image.getOperation(), kernel.getOperation(), padding, strides)), new_shape, dynamic);
}

Tensor sqrt(Tensor& t, bool dynamic=true) {
    DataType dtype = t.getDataType();
    if (dtype < DataType::FLOAT32) {
        std::cout << "ERROR: Data type of operation sqrt must be floating point!" << std::endl;
        assert(false);
    }
    else {
        return Tensor(t, 
            t.getAllocator()->newOperation(
                new SquareRoot(t.getOperation())), dynamic);
    }
}

Tensor cast(Tensor& t, DataType new_dtype, bool dynamic=true) {
    return Tensor(t,
        t.getAllocator()->newOperation(
            new Cast(t.getOperation())), new_dtype, dynamic);
}

Tensor exp(Tensor& t, bool dynamic=true) {
    return Tensor(t, 
        t.getAllocator()->newOperation(
            new Exponential(t.getOperation())), dynamic);
}

} // namespace deeplib
#endif
