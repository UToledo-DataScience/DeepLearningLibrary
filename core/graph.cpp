#include <vector>
#include <stack>
#include <map>
#include "core/tensor.h"
#include "core/utils.h"
#include "core/graph.h"
#include "core/operations.h"
#include "core/allocator.h"

namespace deeplib {

Graph::Graph(Tensor& head, std::vector<Tensor>& leaves, Allocator* allocator) {
    Operation* op_head = head.operation_;
    std::vector<Operation*> op_leaves;

    for (Tensor& t : leaves) {
        t.operate();
        op_leaves.push_back(t.operation_);
    }

    createGraphFromOps(op_head, op_leaves, allocator);
}

void Graph::createGraphFromOps(Operation* head, std::vector<Operation*>& leaves, Allocator* allocator) {
    allocator_ = allocator;

    head->createSelf(head, allocator_);
    heads_.push_back(allocator_->getLatestOperation());

    // Traverse the operation graph
    // and allocate copies of the operations and their buffers.
    std::stack<Operation*> traversal_stack;
    std::stack<Operation*> local_traversal_stack;
    traversal_stack.push(head);
    local_traversal_stack.push(allocator_->getLatestOperation());

    // Working Operation for source graph traversal.
    Operation* traversal_op;
    // Working Operation for this graph during construction.
    Operation* local_op;

    // NOTE: Only supports binary trees. 
    // Graph traversal. For every node, it makes a copy and copies it's connections
    // using the newly allocated nodes.
    while (traversal_stack.size() > 0) {
        traversal_op = traversal_stack.top();
        local_op = local_traversal_stack.top();

        traversal_stack.pop();
        local_traversal_stack.pop();

        Operation* p1 = traversal_op->parent1_;
        Operation* p2 = traversal_op->parent2_;

        if (p1) {
            p1->createSelf(p1, allocator_);
            local_op->parent1_ = allocator_->getLatestOperation();

            if (in<Operation>(p1, leaves) < 0) {
                traversal_stack.push(p1);
                local_traversal_stack.push(allocator_->getLatestOperation());
            }

        }

        if (p2) {
            p2->createSelf(p2, allocator_);
            local_op->parent2_ = allocator_->getLatestOperation();

            if (in<Operation>(p2, leaves) < 0) {
                traversal_stack.push(p2);
                local_traversal_stack.push(allocator_->getLatestOperation());
            }
        }
    }
}

void Graph::traceGraph() {
    std::stack<Operation*> operation_buffer1;
    Operation* op = this->heads_[0];
    operation_buffer1.push(op);

    std::cout << "-------------------------------" << std::endl;
    std::cout << "GRAPH TRACE:" << std::endl;
    while (operation_buffer1.size() > 0) {
        op = operation_buffer1.top();
        std::cout << "Current Operation.type_: " << op->type_ << std::endl;
        operation_buffer1.pop();

        if (op->isNary(2)) {
            if (op->parent1_)
                operation_buffer1.push(op->parent1_);

            if (op->parent2_)
                operation_buffer1.push(op->parent2_);
        }   
        else {
            if (op->parent1_)
                operation_buffer1.push(op->parent1_);
        }   
    }

    std::cout << "-------------------------------" << std::endl;
}

std::vector<Tensor> Graph::graphComputation(std::map<std::string, Tensor> parameters) {
    if (parameters.size() != variables_.size()) {
        std::cout << "Error: Number of given parameters does not match number of Graph Variables." << std::endl;
        assert(false);
    }

    // Fill the Variables in this->variables_ with the given parameters.
    std::map<std::string, Tensor>::iterator it;
    for (it = parameters.begin(); it != parameters.end(); it++) {
        BufferProperties buffer_properties = it->second.getBufferProperties();
        std::string tensor_name = it->first;

        // TODO: A more descriptive error code will probably be more helpful here.
        if (!variables_[tensor_name]->buffer_->checkCompatible(buffer_properties)) {
            std::cout << "Error: Given Tensor " << tensor_name << " incompatible with "
                      << "Graph Variable of same name." << std::endl;
            assert(false);
        }

        variables_[tensor_name]->buffer_->copyData(parameters[tensor_name].getBuffer());
    }

    std::vector<Tensor> results;

    for (int i = 0; i < heads_.size(); i++) {
        Tensor head(heads_[i]);
        head.operate();

        results.push_back(head);
    }

    return results;
}

std::map<std::string, BufferProperties> Graph::getVariableMap() {
    std::map<std::string, BufferProperties> variable_map;

    std::map<std::string, Variable*>::iterator it;
    for (it = variables_.begin(); it != variables_.end(); it++)
        variable_map[it->first] = it->second->buffer_->getProperties();

    return variable_map;
}

void Graph::uproot() {
    for (Operation* op : heads_)
        allocator_->uprootOperation(op);
}

} // namespace deeplib
