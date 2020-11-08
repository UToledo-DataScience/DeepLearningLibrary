#ifndef COMPUTATION_GRAPH
#define COMPUTATION_GRAPH
#include <map>
#include <vector>
#include "core/allocator.h"
#include "core/operations.h"
#include "core/buffer.h"

namespace deeplib {

// Constructed from a head and a set of leaves
// describing where the graph begins and ends.
//
// Construction of a graph copies the given graph and allocates
// new memory for each operation. The allocated memory
// will act as "placeholders", such that whenever the graph
// is called for computation, it takes a given map of values
// to use as Constants to get the calculations going.
//
// Allows for the definition of a mathematical function
// from (a) given Tensor(s) that can be optimized 
// for better performance. Intended use is the
// repeated calculation of a certain variable with
// (possibly) different parameters (located in the Graph
//                                  as Variable Operations).
//
// TODO: Algebraic simplification methods would be both:
//         - Really cool
//       and
//         - Computationally useful
//       for the purposes of this class.
class Graph {
    Allocator* allocator_;

    std::vector<Operation*> heads_;

    std::map<std::string, Variable*> variables_;

  public:
    Graph(Tensor& head, std::vector<Tensor>& leaves, Allocator* allocator);
    // Constructor for when the Graph will consist of the head's entire computation graph.
    Graph(Tensor& head, Allocator* allocator);

    // Creates a new graph that represents the gradient of target w.r.t source.
    Graph gradient(Tensor& target, Tensor& source);

    // Uproots self, deallocating all the memory for this Graph.
    ~Graph();

    void traceGraph();

    // The leaves are the constant parameters of the graph.
    void createGraphFromOps(Operation* head, std::vector<Operation*>& leaves, Allocator* allocator);

    void createGraphFromOps(Operation* head, Allocator* allocator);

    // Numerically calculates the function described by this graph
    // given a set of values to use as Constants.
    //
    // parameters is a map of Variable names to Buffer pointers
    std::vector<Tensor> graphComputation(std::map<std::string, Tensor> parameters);

    std::map<std::string, BufferProperties> getVariableMap();

    void uproot();
};

} // namespace deeplib

#endif
