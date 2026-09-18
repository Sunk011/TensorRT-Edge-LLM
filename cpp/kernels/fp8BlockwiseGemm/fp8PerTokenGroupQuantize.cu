/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fp8PerTokenGroupQuantize.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kGroupSize = kFp8QuantGroupSize; // 128
constexpr int32_t kThreads = 128;                  // one thread per element in a group
constexpr float kFp8E4M3Max = 448.0f;              // float_e4m3fn finite max

//! One CUDA block quantizes one (token, 128-element-group). Grid is 1-D over
//! ``M * groupsPerRow`` so it is not bounded by grid.y's 65535 limit for long
//! prefills. The group amax is reduced across the 128-thread block.
__global__ void fp8PerTokenGroupQuantizeKernel(__half const* __restrict__ input,
    __nv_fp8_storage_t* __restrict__ outputQ, float* __restrict__ outputScale, int32_t K, int32_t groupsPerRow)
{
    int const flatGroup = blockIdx.x;
    int const row = flatGroup / groupsPerRow;
    int const groupIdx = flatGroup - row * groupsPerRow;
    int const tid = threadIdx.x;

    int const kBase = groupIdx * kGroupSize + tid;
    float const x = __half2float(input[static_cast<size_t>(row) * K + kBase]);

    // Block-wide amax over the 128 group elements (4 warps).
    float amax = fabsf(x);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }

    constexpr int32_t kWarps = kThreads / 32;
    __shared__ float warpMax[kWarps];
    __shared__ float groupMax;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    if (lane == 0)
    {
        warpMax[warp] = amax;
    }
    __syncthreads();
    if (tid == 0)
    {
        float m = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            m = fmaxf(m, warpMax[w]);
        }
        groupMax = m;
    }
    __syncthreads();

    float const groupAmax = groupMax;
    // Stored scale matches the reference recipe (amax / 448). The quantize
    // multiplier is the reciprocal, guarded so an all-zero group maps to 0
    // instead of 0/0 = NaN.
    float const storedScale = groupAmax / kFp8E4M3Max;
    float const quantScale = (groupAmax > 0.f) ? (kFp8E4M3Max / groupAmax) : 0.f;

    outputQ[static_cast<size_t>(row) * K + kBase] = __nv_cvt_float_to_fp8(x * quantScale, __NV_SATFINITE, __NV_E4M3);
    if (tid == 0)
    {
        outputScale[static_cast<size_t>(row) * groupsPerRow + groupIdx] = storedScale;
    }
}

} // namespace

void launchFp8PerTokenGroupQuantize(
    void const* input, void* outputQ, void* outputScale, int32_t M, int32_t K, cudaStream_t stream)
{
    int32_t const groupsPerRow = K / kGroupSize;
    if (M <= 0 || groupsPerRow <= 0)
    {
        return;
    }
    int32_t const totalGroups = M * groupsPerRow;
    fp8PerTokenGroupQuantizeKernel<<<totalGroups, kThreads, 0, stream>>>(static_cast<__half const*>(input),
        static_cast<__nv_fp8_storage_t*>(outputQ), static_cast<float*>(outputScale), K, groupsPerRow);
}

} // namespace kernel
} // namespace trt_edgellm
