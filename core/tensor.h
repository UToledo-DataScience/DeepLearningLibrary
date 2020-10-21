#ifndef TENSOR
#define TENSOR
#include <cassert>
#include <vector>
#include <memory>
#include "core/buffer.h"
#include "core/operations.h"
#include "core/data_types.h"

namespace deeplib {

class Tensor {
    uint32_t children_;
    std::vector<int>* shape_;

    Buffer* buffer_;
    Allocator* allocator_;

    DataType dtype_;

    Operation* operation_;

    void incrChildren() { children_++; }

  public:
    Tensor(std::vector<int> new_shape, DataType dt, Allocator* a);

    // Fresh tensor initialized using a 1D set of values.
    // NOTE: This will have to be changed.
    //       Tensors initialized from a set of values will have to happen
    //       some other way. Eigen tensors/matrices?
    Tensor(std::vector<int> values, std::vector<int> s, Allocator* a);

    // Tensor constructed from a binary operation. The operation given
    // is what this tensors operation will be.
    Tensor(Tensor& t1, Tensor& t2, Operation* op);

    // Forced allocation of a new Buffer using a new shape. Currently used for matrix operations
    // which yield a differently shaped output.
    Tensor(Tensor& t1, Tensor& t2, Operation* op, std::vector<int> new_shape);

    // Tensor constructed from a unary operation.
    Tensor(Tensor& t, Operation* op);

    // Constructor for implicit casts from operations.
    Tensor(Tensor& t, Operation* op, DataType new_dtype);

    // Copy constructor.
    Tensor(Tensor& t);

    ~Tensor();

    bool isConstant(Operation* op);

    bool isNary(Operation* op, int n);

    // Operates the tensor, bringing the data in the buffer up to speed
    // at the current operation. 
    void operate();

    // Deallocate this tensor and ALL of it's ancestors.
    // Once this is called, all affected tensors will be unusable.
    //
    // TODO: What to do about orphaned operations?
    void uproot();

    // Getters and setters.

    std::vector<int>& getShape();

    uint64_t getSize();

    uint32_t getChildren();

    Operation* getOperation();

    DataType getDataType();

    Allocator* getAllocator();

    Buffer* getBuffer();

    void setBuffer(Buffer* buf);

    void setName(std::string name);

    void print(bool linear=false);
};

} // namespace deeplib

#endif
