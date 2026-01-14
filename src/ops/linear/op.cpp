#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight,bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(),bias->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous()&& bias->isContiguous(), "linear: all tensors must be contiguous.");
    ASSERT(out->ndim()==2 && in->ndim()==2 && weight->ndim()==2 && bias->ndim()==1,"linear:out-2D,in-2D,weight-2D,bias-1D");
    ASSERT(in->shape()[1]==weight->shape()[1] && out->shape()[0]==in->shape()[0] && out->shape()[1]==weight->shape()[0],"linear:dim match");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(),out->dtype(), out->shape()[0],out->shape()[1],in->shape()[1]);
    }    
}
} // namespace llaisys::ops
