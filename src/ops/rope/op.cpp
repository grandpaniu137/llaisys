#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // Only support contiguous inputs with same shape for now.
   // CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), pos_ids->dtype());
    ASSERT(out->isContiguous() && in->isContiguous(), "rope:out and in must be contiguous.");

    // always support cpu calculation
    // if (c->deviceType() == LLAISYS_DEVICE_CPU) {
    //     return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    // }    
}
} // namespace llaisys::ops
