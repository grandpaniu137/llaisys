#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Add: all tensors must be contiguous.");
    ASSERT(out->ndim()==2 && index->ndim()==1 && weight->ndim()==2,"embedding:out-2D,index-1D,weight-2D");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->numel(),weight->shape()[1]);
    }
}
} // namespace llaisys::ops
