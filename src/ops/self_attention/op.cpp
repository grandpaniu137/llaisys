#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k,v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(),v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous()&&v->isContiguous(), "self_attention: all tensors must be contiguous.");
    ASSERT(attn_val->ndim()==3,"self_attention:3D");
    ASSERT(attn_val->shape()[0]==q->shape()[0]&&attn_val->shape()[1]==q->shape()[1]&&attn_val->shape()[2]==v->shape()[2]&&
        q->shape()[2]==k->shape()[2]&&k->shape()[0]==v->shape()[0]&&k->shape()[1]==v->shape()[1],"dim match");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),scale,q->shape()[0],q->shape()[1],v->shape()[2],
                                    k->shape()[0],k->shape()[1],k->shape()[2]);
    }    
}
} // namespace llaisys::ops
