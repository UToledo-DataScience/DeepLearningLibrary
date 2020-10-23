#include <vector>
#include "core/graph.h"
#include "core/operations.h"
#include "core/allocator.h"

namespace deeplib {

Graph::Graph(Operation* head, std::vector<Operation*> leaves, Allocator* new_allocator) {
    allocator_ = new_allocator;
    heads_.append(head->createSelf());

    // Traverse the operation graph
    // and allocate copies of the operations and their buffers.
    std::stack<Operation*> traversal_stack;
    Operation*
