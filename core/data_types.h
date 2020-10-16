#ifndef DATA_TYPES
#define DATA_TYPES

namespace deeplib {

enum class DataType {
    UINT8 = 0,
    INT8,
    _,

    UINT16,
    INT16,
    __,

    UINT32,
    INT32,
    FLOAT32,

    UINT64,
    INT64,
    FLOAT64,

    BOOL
};

} // namespace deeplib

#endif
