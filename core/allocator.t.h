#include "core/allocator.h"

namespace deeplib {

template <typename AlDType>
void* Allocator::allocate(uint64_t count) {
    void* data = calloc(count+5, sizeof(AlDType));

    uint64_t newly_allocated = count * sizeof(AlDType);
    bytes_allocated += newly_allocated;
    bytes_currently_allocated += newly_allocated;

    return data;
}

} // namespace deeplib
