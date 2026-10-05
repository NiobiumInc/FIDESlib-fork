//
// Created by carlosad on 14/09/25.
//
// The implementation moved to api/engine/SparseEncapsulation.{h,cpp} on
// 2026-09-02: it is pure host OpenFHE code, and living in src/ (CUDA-only)
// meant a CPU-backend client could not generate the SPARSE_ENCAPSULATED
// switching keys at all. This header stays as a shim so CUDA sources and tests
// keep their include path.
//
#ifndef PARAMETERSWITCH_CUH
#define PARAMETERSWITCH_CUH

#include "engine/SparseEncapsulation.h"

#endif // PARAMETERSWITCH_CUH
