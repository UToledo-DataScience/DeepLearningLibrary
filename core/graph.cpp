#include <vector>
#include <stack>
#include <map>
#include "core/tensor.h"
#include "core/utils.h"
#include "core/graph.h"
#include "core/operations.h"
#include "core/allocator.h"

namespace deeplib {

// TODO: UNTESTED
Graph::Graph(Operation* head, std::vector<Operation*> leaves, Allocator* new_allocator) {
    allocator_ = new_allocator;

    head->createSelf(head, allocator_);
    heads_.push_back(allocator_->getLatestOperation());

    // Traverse the operation graph
    // and allocate copies of the operations and their buffers.
    std::stack<Operation*> traversal_stack;
    std::stack<Operation*> local_traversal_stack;
    traversal_stack.push(head);

    // Working Operation for source graph traversal.
    Operation* traversal_op;
    // Working Operation for this graph during construction.
    Operation* local_op = allocator_->getLatestOperation();

    // NOTE: Only supports binary trees. 
    // Graph traversal. For every node, it makes a copy and copies it's connections
    // using the newly allocated nodes. Upon encountering a leaf (i.e. constant parameter),
    // it stops traversal down that branch.
    //
    // This feels very inefficient.
    // TODO: Refactor? A better organizational method for feeding constant parameters
    //       to this graph seems like it should be in order.
    while (traversal_stack.size() > 0) {
        traversal_op = traversal_stack.top();
        local_op = local_traversal_stack.top();

        traversal_stack.pop();
        local_traversal_stack.pop();

        Operation* latest_local_op;

        // Is the check for current_op->parent1_ necessary? Shouldn't be, but
        // {TODO} it needs to be checked.
        if (traversal_op->parent1_) {
            traversal_op->parent1_->createSelf(traversal_op->parent1_, allocator_);
            latest_local_op = allocator_->getLatestOperation();
            local_op->parent1_ = latest_local_op;

            // If the parent1_ isn't a leaf, keep traversing this branch.
            if (!in<Operation>(traversal_op->parent1_, leaves)) {
                traversal_stack.push(traversal_op->parent1_);
                local_traversal_stack.push(latest_local_op);

                local_op = latest_local_op;
            }
            // Variables will only be leaves if included.
            // If the latest operation is a Variable, add it to the list.
            else if (!latest_local_op->type_.compare("variable"))
                variables_[latest_local_op->name_] = dynamic_cast<Variable*>(latest_local_op);
        }

        // A repeat of the above for parent2_.
        if (traversal_op->parent2_) {
            traversal_op->parent2_->createSelf(traversal_op->parent2_, allocator_);
            latest_local_op = allocator_->getLatestOperation();
            local_op->parent2_ = latest_local_op;

            if (!in<Operation>(traversal_op->parent2_, leaves)) {
                traversal_stack.push(traversal_op->parent2_);
                local_traversal_stack.push(latest_local_op);

                local_op = latest_local_op;
            }
            else if (!latest_local_op->type_.compare("variable"))
                variables_[latest_local_op->name_] = dynamic_cast<Variable*>(latest_local_op);
        }
    }
}

// TODO: UNTESTED
std::vector<Tensor> Graph::graphComputation(std::map<std::string, Tensor> parameters) {
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

// TODO: UNTESTED
std::map<std::string, BufferProperties> Graph::getVariableMap() {
    std::map<std::string, BufferProperties> variable_map;

    std::map<std::string, Variable*>::iterator it;
    for (it = variables_.begin(); it != variables_.end(); it++)
        variable_map[it->first] = it->second->buffer_->getProperties();

    return variable_map;
}

} // namespace deeplib
