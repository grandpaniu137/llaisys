#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t row,size_t col,float eps) {
    for(size_t i=0;i<row;i++){
        float temp=0;
        for(size_t j=0;j<col;j++){
            if constexpr (std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                temp+=pow((llaisys::utils::cast<float>(in[i*col+j])),2);
            }else{
                temp+=pow(in[i*col+j],2);
            }
        }
        temp=sqrt(temp/col+eps);
        for(size_t j=0;j<col;j++){
            if constexpr (std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                out[i*col+j]=llaisys::utils::cast<T>((llaisys::utils::cast<float>(weight[j])*(llaisys::utils::cast<float>(in[i*col+j])))/temp);
            }else{
                out[i*col+j]=(weight[j]*in[i*col+j])/temp;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t row,size_t col,float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), row,col,eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), row,col,eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), row,col,eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
