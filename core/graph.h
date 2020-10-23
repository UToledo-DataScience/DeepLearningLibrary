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
class Graph {
    Allocator* allocator_;

    std::vector<Operation*> head_;

    std::vector<Operation*> constants_;

  public:

    Graph(Operation* head, std::vector<Operation*> leaves, Allocator* new_allocator);

    // Numerically calculates the function described by this graph
    // given a set of values to use as Constants.
    Tensor graph_computation(std::map<Constant*, Buffer*> constants_values);
};

} // namespace deeplib

#endif
