// DEBUG BRANCH ONLY. Routes the event records and stream waits of the file that includes this header
// through the table-owner log (TableOwner.cuh), so a report can show which stream-order edges existed
// between the free of a table's memory and its reuse on another stream.
//
// Include it AFTER every other header of the file: from here on, cudaEventRecord,
// cudaEventRecordWithFlags and cudaStreamWaitEvent are macros. The wrappers (CudaUtils.cu) call the
// real functions as `(cudaEventRecord)(...)`, which a function-like macro does not expand.
#pragma once

#include <cuda_runtime.h>

cudaError_t TableOwnerEventRecord(cudaEvent_t e, cudaStream_t s = 0);
cudaError_t TableOwnerEventRecordWithFlags(cudaEvent_t e, cudaStream_t s = 0, unsigned int flags = 0);
cudaError_t TableOwnerStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags = 0);

#define cudaEventRecord(...) TableOwnerEventRecord(__VA_ARGS__)
#define cudaEventRecordWithFlags(...) TableOwnerEventRecordWithFlags(__VA_ARGS__)
#define cudaStreamWaitEvent(...) TableOwnerStreamWaitEvent(__VA_ARGS__)
