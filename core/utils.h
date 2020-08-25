#ifndef UTILS
#define UTILS
#include <vector>

// return true if vectors are element-wise equal
// otherwise false
bool compare(std::vector<int>& v1, std::vector<int>& v2) {
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
#endif
