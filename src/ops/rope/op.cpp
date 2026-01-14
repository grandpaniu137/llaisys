#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(),LLAISYS_DTYPE_I64);
    ASSERT(out->isContiguous() && in->isContiguous(), "rope:out and in must be contiguous.");
    ASSERT(out->ndim()==3 && in->ndim()==3 && pos_ids->ndim()==1,"rope:out-3D,in-3D,pos_ids-1D");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(),theta,in->shape()[0],in->shape()[1],in->shape()[2]);
    }    
}
} // namespace llaisys::ops
