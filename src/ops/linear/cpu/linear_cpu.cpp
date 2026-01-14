#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight,const T *bias,size_t row,size_t col,size_t mid) {
   for(size_t i=0;i<row;i++){
        for(size_t j=0;j<col;j++){
            float temp=0;
            for(size_t k=0;k<mid;k++){
                if constexpr(std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                    temp+=(llaisys::utils::cast<float>(in[i*mid+k])) *(llaisys::utils::cast<float>(weight[j*mid+k]));  
                }else{
                    temp+=in[i*mid+k]*weight[j*mid+k];
                }
            }
            if(bias != nullptr){
                if constexpr(std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                    temp+=llaisys::utils::cast<float>(bias[j]);
                }else{
                    temp+=bias[j];
                }
            }
            if constexpr(std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                out[i*col+j]=llaisys::utils::cast<T>(temp);
            }else{
                out[i*col+j]=temp;
            }
        }
   }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,const std::byte *bias,llaisysDataType_t type, size_t row,size_t col,size_t mid) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight),
        reinterpret_cast<const float *>(bias),row,col,mid);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight),reinterpret_cast<const llaisys::bf16_t *>(bias),row,col,mid);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight),reinterpret_cast<const llaisys::fp16_t *>(bias), row,col,mid);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
