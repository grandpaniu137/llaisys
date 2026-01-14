#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "rms_norm: all tensors must be contiguous.");
    ASSERT(out->ndim()==2 && in->ndim()==2 && weight->ndim()==1 ,"rms_norm:out-2D,in-2D,weight-1D");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0],in->shape()[1],eps);
    }    
}
} // namespace llaisys::ops
