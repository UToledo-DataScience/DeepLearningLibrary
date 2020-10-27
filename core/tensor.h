#ifndef TENSOR
#define TENSOR
#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include "core/buffer.h"
#include "core/operations.h"
#include "core/data_types.h"

namespace deeplib {

// Is this necessary? At the moment it seems so...
// see Graph::graphComputation, specifically the 
// std::map<std::string, TensorProperties> parameter.
// Passing a straight Tensor has the compiler disagree because
// it's an incomplete type.
struct TensorProperties {
    uint32_t children_;

    Buffer* buffer_;
    Allocator* allocator_;

    DataType dtype_;

    Operation* operation_;

    bool dynamic_;
};

// TODO: This (and probably others) need better organization regarding 
//       public/protected/private design.
class Tensor {
    friend class Graph;

    uint32_t children_;

    Buffer* buffer_;
    Allocator* allocator_;

    DataType dtype_;

    Operation* operation_;

    bool dynamic_;

    // If a Tensor needs created from an existing Operation.
    // NOTE: Assumes that the Tensor has no children.
    Tensor(Operation* op);

    Buffer* getBuffer() const;

    void setBuffer(Buffer* buf);

    void incrChildren() { children_++; }

  public:
    Tensor();

    Tensor(std::vector<int> new_shape, DataType dt, Allocator* a);

    // Fresh tensor initialized using a 1D set of values.
    // NOTE: This will have to be changed.
    //       Tensors initialized from a set of values will have to happen
    //       some other way. Eigen tensors/matrices?
    Tensor(std::vector<int> values, std::vector<int> s, Allocator* a, bool variable=false, std::string name="");

    // Tensor constructed from a binary operation. The operation given
    // is what this tensors operation will be.
    Tensor(Tensor& t1, Tensor& t2, Operation* op, bool dynamic);

    // Forced allocation of a new Buffer using a new shape. Currently used for matrix operations
    // which yield a differently shaped output.
    Tensor(Tensor& t1, Tensor& t2, Operation* op, std::vector<int> new_shape, bool dynamic);

    // Tensor constructed from a unary operation.
    Tensor(Tensor& t, Operation* op, bool dynamic);

    // Constructor for implicit casts from operations.
    Tensor(Tensor& t, Operation* op, DataType new_dtype, bool dynamic);

    // Copy constructor.
    Tensor(const Tensor& t);

    ~Tensor();

    // Traverses the Graph and prints the Operation node type as it goes.
    void traceGraph();

    // Operates the tensor, bringing the data in the buffer up to speed
    // at the current operation. 
    void operate();

    // Calculates the gradient of this tensor
    // with respect to target.
    //Tensor gradient(Tensor& target);

    // Deallocate this tensor and ALL of it's ancestors.
    // Once this is called, all affected tensors will be unusable.
    //
    // TODO: What to do about orphaned operations?
    void uproot();

    // Getters and setters.

    std::vector<int>& getShape() const;

    uint64_t getSize() const;

    uint32_t getChildren() const;

    Operation* getOperation() const;

    DataType getDataType() const;

    Allocator* getAllocator() const;

    void setName(std::string name);

    void print(bool linear=false);

    TensorProperties getTensorProperties() const;

    BufferProperties getBufferProperties() const;
};

} // namespace deeplib

#endif
