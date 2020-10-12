#ifndef OP_FUNCTIONS
#define OP_FUNCTIONS
#include "core/tensor.h"
#include "core/operations.h"

namespace deeplib {

// NOTE: It is assumed that the allocators of t1 and t2 are the same.
//       this may change in the future.

Tensor add(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Addition(t1.getOperation(), t2.getOperation())));
}

Tensor sub(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Subtraction(t1.getOperation(), t2.getOperation())));
}

Tensor power(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());
    
    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Power(t1.getOperation(), t2.getOperation())));
}

Tensor multiply(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    if (&t1 == &t2) {
       return power(t1, t2);
    }

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Multiplication(t1.getOperation(), t2.getOperation())));
}

Tensor divide(Tensor& t1, Tensor& t2) {
    assert(t1.getDataType() == t2.getDataType());

    return Tensor(t1, t2,
        t1.getAllocator()->newOperation(
            new Division(t1.getOperation(), t2.getOperation())));
}

// Inputs must be at least 2D. Inputs of higher rank
// will only be considered for their last two dimensions.
//
// Shape is assumed to be in format [..., rows, columns]
Tensor matmul(Tensor& t1, Tensor& t2) {
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
            new MatrixMultiplication(t1.getOperation(), t2.getOperation())), new_shape);
}

// TODO: padding, strides, dilation_rate
Tensor conv2d(Tensor& image, Tensor& kernel) {
    assert(image.getDataType() == kernel.getDataType());
    std::vector<int>& image_shape = image.getShape();
    std::vector<int>& kernel_shape = kernel.getShape();

    std::vector<int> new_shape;

    int kernel_offset_y = std::ceil(static_cast<float>(kernel_shape[0]) / 2) - 1;
    int kernel_offset_x = std::ceil(static_cast<float>(kernel_shape[1]) / 2) - 1;

    assert(image_shape.size() >= 2 && kernel_shape.size() == 2);
    for (int i = 0; i < image_shape.size()-2; i++)
        new_shape.push_back(image_shape[i]);

    new_shape.push_back(image_shape[image_shape.size()-2] - kernel_shape[0] + 1);
    new_shape.push_back(image_shape[image_shape.size()-1] - kernel_shape[1] + 1);

    return Tensor(image, kernel,
        image.getAllocator()->newOperation(
            new Convolution2D(image.getOperation(), kernel.getOperation())), new_shape);
}

Tensor sqrt(Tensor& t) {
    DataType dtype = t.getDataType();
    if (dtype < DataType::FLOAT32) {
        std::cout << "ERROR: Data type of operation sqrt must be floating point!" << std::endl;
        assert(false);
    }
    else {
        return Tensor(t, 
            t.getAllocator()->newOperation(
                new SquareRoot(t.getOperation())));
    }
}

Tensor cast(Tensor& t, DataType new_dtype) {
    return Tensor(t,
        t.getAllocator()->newOperation(
            new Cast(t.getOperation())), new_dtype);
}

Tensor exp(Tensor& t) {
    return Tensor(t, 
        t.getAllocator()->newOperation(
            new Exponential(t.getOperation())));
}

} // namespace deeplib
#endif
