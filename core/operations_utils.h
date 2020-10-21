#ifndef OPERATIONS_UTILS
#define OPERATIONS_UTILS
#include "core/buffer.h"
#include "core/data_types.h"

namespace deeplib {

// Feed an Operation type (not Operation itself though) in here along with the buffers
// to avoid unnecessary repeating of the switch statements.
// 
// A better method than the previous in avoiding code bloat (I hope).
template <class Op>
void compTemplateChoice(Op* op, Buffer* b1, Buffer* b2, DataType dtype) {
    switch (dtype) {
      case DataType::UINT8:
        op->template compute<uint8_t>(b1, b2);
        return;

      case DataType::UINT16:
        op->template compute<uint16_t>(b1, b2);
        return;

      case DataType::UINT32:
        op->template compute<uint32_t>(b1, b2);
        return;

      case DataType::UINT64:
        op->template compute<uint64_t>(b1, b2);
        return;

      case DataType::INT8:
        op->template compute<int8_t>(b1, b2);
        return;

      case DataType::INT16:
        op->template compute<int16_t>(b1, b2);
        return;

      case DataType::INT32:
        op->template compute<int32_t>(b1, b2);
        return;

      case DataType::INT64:
        op->template compute<int64_t>(b1, b2);
        return;
            
      case DataType::FLOAT32:
        op->template compute<float>(b1, b2);
        return;

      case DataType::FLOAT64:
        op->template compute<double>(b1, b2);
        return;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

// Overloaded for unary operations.
template <class Op>
void compTemplateChoice(Op* op, Buffer* b1, DataType dtype) {
    switch (dtype) {
      case DataType::UINT8:
        op->template compute<uint8_t>(b1);
        return;

      case DataType::UINT16:
        op->template compute<uint16_t>(b1);
        return;

      case DataType::UINT32:
        op->template compute<uint32_t>(b1);
        return;

      case DataType::UINT64:
        op->template compute<uint64_t>(b1);
        return;

      case DataType::INT8:
        op->template compute<int8_t>(b1);
        return;

      case DataType::INT16:
        op->template compute<int16_t>(b1);
        return;

      case DataType::INT32:
        op->template compute<int32_t>(b1);
        return;

      case DataType::INT64:
        op->template compute<int64_t>(b1);
        return;
            
      case DataType::FLOAT32:
        op->template compute<float>(b1);
        return;

      case DataType::FLOAT64:
        op->template compute<double>(b1);
        return;

      default:
        std::cout << "ERROR: bad data type!" << std::endl;
        assert(false);
    }
}

} //namespace deeplib

#endif
