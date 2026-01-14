#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_idx, float theta,size_t seqlen,size_t nhead,size_t d) {
    int stride_seq=d*nhead;
    size_t half=d/2;
    for(size_t i=0;i<seqlen;i++){
        int64_t pos=pos_idx[i];
        int seq_offset=i*stride_seq;
        for(size_t j=0;j<nhead;j++){
            for(size_t k=0;k<half;k++){
                float phi=(float)pos*(std::pow(theta,-2.0*(float)k/(float)d));
                if constexpr(std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                    float a=llaisys::utils::cast<float>(in[seq_offset+j*d+k]);
                    float b=llaisys::utils::cast<float>(in[seq_offset+j*d+k+half]);
                    out[seq_offset+j*d+k]=llaisys::utils::cast<T>(a*(std::cos(phi))-b*(std::sin(phi)));
                    out[seq_offset+j*d+k+half]=llaisys::utils::cast<T>(b*(std::cos(phi))+a*(std::sin(phi)));
                }else{
                    float a=in[seq_offset+j*d+k];
                    float b=in[seq_offset+j*d+k+half];
                    out[seq_offset+j*d+k]=a*(std::cos(phi))-b*(std::sin(phi));
                    out[seq_offset+j*d+k+half]=b*(std::cos(phi))+a*(std::sin(phi));
                }   
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_idx, llaisysDataType_t type,float theta,size_t seqlen,size_t nhead,size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t*>(pos_idx),theta,seqlen,nhead,d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const int64_t*>(pos_idx), theta,seqlen,nhead,d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_idx), theta,seqlen,nhead,d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
