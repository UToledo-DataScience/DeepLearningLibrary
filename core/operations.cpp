#include <iostream>
#include <cmath>
#include "core/operations.h"
#include "core/operations_utils.h"

using std::string;

namespace deeplib {

// TODO: Please rethink how Operations are handled. Continuing in this manner
//       will be the death to any sort of maintainability if we don't clean up.

//-----------------------------------\\
// class Operation;                  \\
//-----------------------------------\\

// Abstract operation graph node class.
//
// Each Operation has two key functions:
//   - Operation.derive()
//   - Operation.operate()
//
// derive() is the function corresponding to the
// derivative of that specific operation.
// e.g. Multiplication.derive == d/dx(x * x) == x*1 + 1*x == product rule
//
// operate() is a recursive used to actually enact the arithmetic
// defined by the operation.
// e.g. Multiply.operate() == x * y,
//      where x == Operation.parent1_ and y == Operation.parent2_
//
// It should be noted that Constant.operate() retrieves the actual value,
// acting as the end condition to the recursive function
// as newly created Tensor<T> objects have a Constant as their operation. // TODO: review this last line
//
// This is never to be used directly.
Operation::Operation() {
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->buffer_ = nullptr;
}

void Operation::setName(std::string name) {
    this->name_ = name;
}

bool Operation::isConstant() {
    return !this->type_.compare("constant");
}

bool Operation::isNary(int n) {
    return this->ary_ == n;
}

//-----------------------------------\\
// class Addition;                   \\
//-----------------------------------\\

// Operation graph node for element-wise multiplication.
Addition::Addition(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_addition";
    this->type_ = "addition";
    this->ary_ = 2;
}

Addition::Addition(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_addition"; // Are we ever going to use this->name_?
    this->type_ = "addition";
    this->ary_ = 2;
}

void Addition::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Addition::getBuffer() { return this->buffer_; }

void Addition::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Addition::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Addition>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Addition::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Addition>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Addition::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Addition::operate called on a binary function!" << std::endl;
    assert(false);
}

void Addition::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Addition(source, allocator));
}

//-----------------------------------\\
// class Subtraction;                \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Subtraction::Subtraction(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_subtraction";
    this->type_ = "subtraction";
    this->ary_ = 2;
}

Subtraction::Subtraction(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_subtraction"; // Are we ever going to use this->name_?
    this->type_ = "subtraction";
    this->ary_ = 2;
}

void Subtraction::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Subtraction::getBuffer() { return this->buffer_; }

void Subtraction::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Subtraction::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Subtraction>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Subtraction::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Subtraction>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Subtraction::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Subtraction::operate called on a binary function!" << std::endl;
    assert(false);
}

void Subtraction::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Subtraction(source, allocator));
}

//-----------------------------------\\
// class Multiplication;             \\
//-----------------------------------\\

// Operation graph node for element-wise multiplication.
Multiplication::Multiplication(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->gradient_computation_buffer_ = nullptr;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_multiplication";
    this->type_ = "multiplication";
    this->ary_ = 2;
}

Multiplication::Multiplication(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_multiplication"; // Are we ever going to use this->name_?
    this->type_ = "multiplication";
    this->ary_ = 2;
}

void Multiplication::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Multiplication::getBuffer() { return this->buffer_; }

void Multiplication::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Multiplication::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Multiplication>(this, b1, b2, dtype);

    this->computed_ = true;
    return this->buffer_;
}

void Multiplication::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Multiplication>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Multiplication::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Multiplication::operate called on a binary function!" << std::endl;
    assert(false);
}

void Multiplication::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Multiplication(source, allocator));
}

/*Buffer* Multiplication::gradient(Buffer* b1, Buffer* b2, Buffer* grad1, Buffer* grad2) {
    if (!this->computed_)
        this->buffer_ = this->operate();

    this->gradient_computation_buffer_ = this->allocator_->newBuffer(new Buffer(this->buffer_, false));
    this->gradient_output_buffer_ = this->allocator_->newBuffer(new Buffer(this->buffer_, false));

    DataType dtype = b1->getDataType();

    gradTemplateChoice<Multiplication>(this, b1, b2, grad1, grad2, dtype);
}*/

//-----------------------------------\\
// class Division;                   \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Division::Division(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_division";
    this->type_ = "division";
    this->ary_ = 2;
}

Division::Division(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_division"; // Are we ever going to use this->name_?
    this->type_ = "division";
    this->ary_ = 2;
}

void Division::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Division::getBuffer() { return this->buffer_; }

void Division::derive() {}

// NOTE: Broadcasting only supported for constants.
//
// Single-threaded approach.
//
// Element-wise multiplication - no shape change.
Buffer* Division::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Division>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Division::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Division>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Division::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Division::operate called on a binary function!" << std::endl;
    assert(false);
}

void Division::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Division(source, allocator));
}

//-----------------------------------\\
// class MatrixMultiplication;       \\
//-----------------------------------\\

// Operation graph node for element-wise division.
MatrixMultiplication::MatrixMultiplication(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_matrix_multiplication";
    this->type_ = "matrix_multiplication";
    this->ary_ = 2;
}

MatrixMultiplication::MatrixMultiplication(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_matrix_multiplication"; // Are we ever going to use this->name_?
    this->type_ = "matrix_multiplication";
    this->ary_ = 2;
}

void MatrixMultiplication::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* MatrixMultiplication::getBuffer() { return this->buffer_; }

void MatrixMultiplication::derive() {}

Buffer* MatrixMultiplication::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<MatrixMultiplication>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void MatrixMultiplication::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<MatrixMultiplication>(this, b1, b2, dtype);
    this->computed_ = true;
}

void MatrixMultiplication::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function MatrixMultiplication::operate called on a binary function!" << std::endl;
    assert(false);
}

void MatrixMultiplication::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new MatrixMultiplication(source, allocator));
}

//-----------------------------------\\
// class Convolution2D;              \\
//-----------------------------------\\

// Operation graph node for element-wise division.
Convolution2D::Convolution2D(Operation* p1, Operation* p2, std::string padding, int (&strides)[2]) {
    this->computed_ = false;
    this->padding_ = padding;
    this->strides_[0] = strides[0];
    this->strides_[1] = strides[1];
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_convolution2d";
    this->type_ = "convolution2d";
    this->ary_ = 2;
}

Convolution2D::Convolution2D(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    // TODO
    //this->padding_ = source->padding_;
    //this->strides_[0] = source->strides_[0];
    //this->strides_[1] = source->strides_[1];
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_convolution2d"; // Are we ever going to use this->name_?
    this->type_ = "convolution2d";
    this->ary_ = 2;
}

void Convolution2D::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Convolution2D::getBuffer() { return this->buffer_; }

void Convolution2D::derive() {}

Buffer* Convolution2D::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Convolution2D>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Convolution2D::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Convolution2D>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Convolution2D::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Convolution2D::operate called on a binary function!" << std::endl;
    assert(false);
}

void Convolution2D::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Convolution2D(source, allocator));
}

//-----------------------------------\\
// class Power;                      \\
//-----------------------------------\\

Power::Power(Operation* p1, Operation* p2) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = p2;
    this->name_ = "unnamed_power";
    this->type_ = "power";
    this->ary_ = 2;
}

Power::Power(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_power"; // Are we ever going to use this->name_?
    this->type_ = "power";
    this->ary_ = 2;
}

void Power::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Power::getBuffer() { return this->buffer_; }

void Power::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Power::operate() {
    this->buffer_->initialize();

    Buffer* b1 = this->parent1_->operate();
    Buffer* b2 = this->parent2_->operate();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Power>(this, b1, b2, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Power::operate(Buffer* b1, Buffer* b2) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Power>(this, b1, b2, dtype);
    this->computed_ = true;
}

void Power::operate(Buffer* b1) {
    std::cout << "Error: Unary overloaded function Power::operate called on a binary function!" << std::endl;
    assert(false);
}

void Power::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Power(source, allocator));
}

//-----------------------------------\\
// class Cast;                       \\
//-----------------------------------\\

Cast::Cast(Operation* p1) {
    this->computed_ = false;
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_cast";
    this->type_ = "cast";
    this->ary_ = 1;
}

Cast::Cast(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_cast"; // Are we ever going to use this->name_?
    this->type_ = "cast";
    this->ary_ = 1;
}

void Cast::setBuffer(Buffer* buf) {
    this->buffer_ = buf;
}

Buffer* Cast::getBuffer() {
    return this->buffer_;
}

void Cast::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Cast::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    DataType dtype = this->buffer_->getDataType();

    compTemplateChoice<Cast>(this, buf, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Cast::operate(Buffer* b1, Buffer* b2) {
    std::cout << "Error: Binary overloaded function Cast::operate called on a unary function!" << std::endl;
    assert(false);
}

void Cast::operate(Buffer* b1) {
    this->buffer_->initialize();

    DataType dtype = this->buffer_->getDataType();

    compTemplateChoice<Cast>(this, b1, dtype);
    this->computed_ = true;
}

void Cast::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Cast(source, allocator));
}

//-----------------------------------\\
// class SquareRoot;                 \\
//-----------------------------------\\

SquareRoot::SquareRoot(Operation* p1, bool promotion) {
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_square_root";
    this->type_ = "square_root";
    this->promotion = promotion;
    this->ary_ = 1;
}

SquareRoot::SquareRoot(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_square_root"; // Are we ever going to use this->name_?
    this->type_ = "square_root";
    this->ary_ = 1;
}

void SquareRoot::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* SquareRoot::getBuffer() { return this->buffer_; }

void SquareRoot::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* SquareRoot::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    //DataType dtype;
    //if (this->promotion)
    //    dtype = DataType::FLOAT32;
    //else
    DataType dtype = this->buffer_->getDataType();

    compTemplateChoice<SquareRoot>(this, buf, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void SquareRoot::operate(Buffer* b1, Buffer* b2) {
    std::cout << "Error: Binary overloaded function SquareRoot::operate called on a unary function!" << std::endl;
    assert(false);
}

void SquareRoot::operate(Buffer* b1) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<SquareRoot>(this, b1, dtype);
    this->computed_ = true;
}

void SquareRoot::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new SquareRoot(source, allocator));
}

//-----------------------------------\\
// class Exponential;                \\
//-----------------------------------\\

Exponential::Exponential(Operation* p1) {
    this->parent1_ = p1;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_exponential";
    this->type_ = "exponential";
    this->ary_ = 1;
}

Exponential::Exponential(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = false;
    this->parent1_ = nullptr;
    this->parent2_ = nullptr;
    this->name_ = "unnamed_exponential"; // Are we ever going to use this->name_?
    this->type_ = "exponential";
    this->ary_ = 1;
}

void Exponential::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Exponential::getBuffer() { return this->buffer_; }

void Exponential::derive() {}

// NOTE: broadcasting not yet supported
//
// element-wise multiplication - no shape change
Buffer* Exponential::operate() {
    this->buffer_->initialize();

    Buffer* buf = this->parent1_->operate();

    DataType dtype = buf->getDataType();

    compTemplateChoice<Exponential>(this, buf, dtype);
    this->computed_ = true;
    return this->buffer_;
}

void Exponential::operate(Buffer* b1, Buffer* b2) {
    std::cout << "Error: Binary overloaded function Exponential::operate called on a unary function!" << std::endl;
    assert(false);
}

void Exponential::operate(Buffer* b1) {
    this->buffer_->initialize();

    DataType dtype = b1->getDataType();

    compTemplateChoice<Exponential>(this, b1, dtype);
    this->computed_ = true;
}

void Exponential::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Exponential(source, allocator));
}

//-----------------------------------\\
// class Constant;                   \\
//-----------------------------------\\

Constant::Constant(Buffer* buf) {
    this->buffer_ = buf;
    this->computed_ = true;
    this->name_ = "unnamed_constant";
    this->type_ = "constant";
    this->ary_ = 0;
}

Constant::Constant(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), true));
    this->computed_ = true;
    this->name_ = "unnamed_constant"; // Are we ever going to use this->name_?
    this->type_ = "constant";
    this->ary_ = 0;
}

void Constant::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Constant::getBuffer() { return this->buffer_; }

void Constant::derive() {
    return;
}

Buffer* Constant::operate() {
    return this->buffer_;
}

void Constant::operate(Buffer* b1, Buffer* b2) {
    std::cout << "Error: Binary overloaded function Constant::operate called on a unary function!" << std::endl;
    assert(false);
}

void Constant::operate(Buffer* b1) {
    this->operate();
    //std::cout << "Error: Unary overloaded function Constant::operate called on a Constant Operator!" << std::endl;
    //assert(false);
}

void Constant::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Constant(source, allocator));
}

//-----------------------------------\\
// class Variable;                   \\
//-----------------------------------\\

Variable::Variable(Buffer* buf) {
    this->buffer_ = buf;
    this->computed_ = true;
    this->name_ = "unnamed_variable";
    this->type_ = "variable";
    this->ary_ = 0;
}

Variable::Variable(Operation* source, Allocator* allocator) {
    this->buffer_ = allocator->newBuffer(new Buffer(source->getBuffer(), false));
    this->computed_ = true;
    this->name_ = "unnamed_variable"; // Are we ever going to use this->name_?
    this->type_ = "variable";
    this->ary_ = 0;
}

void Variable::setBuffer(Buffer* buf) { this->buffer_ = buf; }

Buffer* Variable::getBuffer() { return this->buffer_; }

void Variable::derive() {
    return;
}

Buffer* Variable::operate() {
    return this->buffer_;
}

void Variable::operate(Buffer* b1, Buffer* b2) {
    std::cout << "Error: Binary overloaded function Variable::operate called on a unary function!" << std::endl;
    assert(false);
}

void Variable::operate(Buffer* b1) {
    this->operate();
}

void Variable::createSelf(Operation* source, Allocator* allocator) {
    allocator->newOperation(new Variable(source, allocator));
}

} // namespace deeplib
