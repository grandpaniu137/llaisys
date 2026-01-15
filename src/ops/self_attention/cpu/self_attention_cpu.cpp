#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
float minus_inf=-std::numeric_limits<float>::infinity();

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k,const T *v,float scale,size_t seqlen,size_t nhead,size_t dv,size_t total_len,size_t nkvhead,size_t d) {
    std::vector<float> logits(total_len);
    std::vector<float> weights(total_len);
    size_t group=nhead/nkvhead;

    for(size_t i=0;i<seqlen;i++){
        for(size_t j=0;j<nhead;j++){
            const T* q_s=q+(i*nhead+j)*d;
            size_t kv_idx=j/group;
            float max_logit=minus_inf;
            //A=QK^T*scale
            for(size_t m=0;m<total_len;m++){
                const T* k_s=k+(m*nkvhead+kv_idx)*d;
                float result=0.0;
                for(size_t n=0;n<d;n++){
                    if constexpr (std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                        float q_temp=llaisys::utils::cast<float>(q_s[n]);
                        float k_temp=llaisys::utils::cast<float>(k_s[n]);
                        result+=q_temp*k_temp;
                    }else{
                        result+=q_s[n]*k_s[n];
                    }
                }
                result*=scale;
                //casual
                if(i+total_len-seqlen>=m){
                    logits[m]=result;
                    max_logit=std::max(max_logit,result);
                }else{
                    logits[m]=minus_inf;
                    weights[m]=minus_inf;
                }
            }
            //softmax
            float sum=0.0;
            for(size_t m=0;m<total_len&&i+total_len-seqlen>=m;m++){
                float num=std::exp((float)(logits[m]-max_logit));
                weights[m]=num;
                sum+=num;
            }
            float inv_sum=1.0f/(sum+1e-6f);
            for(size_t m=0;m<total_len&&i+total_len-seqlen>=m;m++){
                weights[m]*=inv_sum;
            }
            std::vector<float> ans(dv,0.0);
            //attn=casualsoftmax(A)V
            for(size_t m=0;m<total_len&&i+total_len-seqlen>=m;m++){
                const T* v_s=v+(m*nkvhead+kv_idx)*dv;
                for(size_t n=0;n<dv;n++){
                    if constexpr (std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                        ans[n]+=weights[m]*llaisys::utils::cast<float>(v_s[n]);
                    }else{
                        ans[n]+=weights[m]*v_s[n];
                    }
                }
            }
            T* out_s=attn_val+(i*nhead+j)*dv;
            for(size_t n=0;n<dv;n++){
                if constexpr (std::is_same_v<T,llaisys::bf16_t>||std::is_same_v<T,llaisys::fp16_t>){
                    out_s[n]=llaisys::utils::cast<T>(ans[n]);
                }else{
                    out_s[n]=ans[n];
                }
            }    
        }
    }
}


namespace llaisys::ops::cpu{
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,const std::byte *v, llaisysDataType_t type,float scale,
                    size_t seqlen,size_t nhead,size_t dv,size_t total_len,size_t nkvhead,size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k),
        reinterpret_cast<const float*>(v),scale,seqlen,nhead,dv,total_len,nkvhead,d);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), 
        reinterpret_cast<const llaisys::bf16_t *>(k),reinterpret_cast<const llaisys::bf16_t *>(v),scale,seqlen,nhead,dv,total_len,nkvhead,d);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), 
        reinterpret_cast<const llaisys::fp16_t *>(k),reinterpret_cast<const llaisys::fp16_t *>(v),scale,seqlen,nhead,dv,total_len,nkvhead,d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
}
}
} // namespace llaisys::ops::cpu
