#ifndef UTILS
#define UTILS
#include <vector>
#include <sstream>
#include <algorithm>
#include "core/data_types.h"

// return true if vectors are element-wise equal
// otherwise false
static bool compare(std::vector<int>& v1, std::vector<int>& v2) {
    if (v1.size() != v2.size()) {
        std::cout << "Warning: mismatching vector sizes in compare()" << std::endl;
        return false;
    }

    for (int i = 0; i < v1.size(); i++) {
        if (v1[i] != v2[i])
            return false;
    }

    return true;
}

template <typename T>
static std::string vecToString(std::vector<T>& vec) {
    try {
        std::stringstream ss;
        ss << "( ";
        for (auto i : vec)
            ss << i << " ";

        ss << ")";

        return ss.str();
    }
    catch(std::exception& e) {
        std::cout << e.what() << std::endl;
        return e.what();
    }
}

// checks for broadcastability between the two vectors
// true for yes, false for no
//
// Each dimension must be one or both of the following:
//   - be equal for both shapes
//   - be 1 for at least one of the shapes
static bool broadcastable(std::vector<int>& v1, std::vector<int>& v2) {
    // check for violation of the above conditions
    int diff = std::max(v1.size(), v2.size()) - std::min(v1.size(), v2.size());
    std::vector<int>* bigger_vec;
    std::vector<int>* smaller_vec;

    // else block covers the case where v1.size() == v2.size()
    // as well as if v2.size() > v1.size()
    if (v1.size() > v2.size()) {
        bigger_vec = &v1;
        smaller_vec = &v2;
    }
    else {
        bigger_vec = &v2;
        smaller_vec = &v1;
    }

    for (int i = bigger_vec->size()-1; i-diff > -1; i--) {
        if (bigger_vec->at(i) != smaller_vec->at(i-diff) && 
            bigger_vec->at(i) != 1 && smaller_vec->at(i-diff) != 1) {
            std::cout << "Error: shapes " << vecToString(v1)
                      << " and " << vecToString(v2) << " are incompatible broadcast shapes." << std::endl;

            return false;
        }
    }

    return true;
}

// returns the index at which the element was found
// returns -1 if the element is not in the vector
template <typename T>
static int in(T* element, std::vector<T*>& list) {
    for (int i = 0; i < list.size(); i++) {
        if (element == list[i])
            return i;
    }

    return -1;
}

#endif
