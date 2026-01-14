#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type,float theta,size_t seqlen,size_t nhead,size_t d);
}