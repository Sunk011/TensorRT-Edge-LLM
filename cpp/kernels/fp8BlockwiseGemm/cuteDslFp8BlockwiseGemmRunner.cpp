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

#include "cuteDslFp8BlockwiseGemmRunner.h"

#ifdef CUTE_DSL_FP8_BLOCK_GEMM_ENABLED

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_fp8_block_gemm_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

namespace trt_edgellm::kernel
{
namespace
{

detail::LazyKernelModule<fp8_blockwise_gemm_fp16_Kernel_Module_t> gModule{};

bool isSupportedSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110;
}

template <typename Tensor>
void setTensor(Tensor& tensor, void* data, int32_t dim0, int32_t dim1)
{
    tensor.data = data;
    tensor.dynamic_shapes[0] = dim0;
    tensor.dynamic_shapes[1] = dim1;
    tensor.dynamic_shapes[2] = 1;
    tensor.dynamic_strides[0] = dim1;
    tensor.dynamic_strides[1] = static_cast<int64_t>(dim0) * dim1;
}

bool ensureModule(cudaStream_t stream)
{
    return detail::ensureModuleLoaded<fp8_blockwise_gemm_fp16_Kernel_Module_Load,
        fp8_blockwise_gemm_fp16_Kernel_Module_Unload>(gModule, "fp8_blockwise_gemm_fp16", stream);
}

} // namespace

bool fp8BlockwiseCuteDslGemmSupported()
{
    int32_t device{};
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
        return false;
    }
    return isSupportedSm(properties.major * 10 + properties.minor);
}

bool fp8BlockwiseCuteDslGemmCanImplement(int32_t M, int32_t N, int32_t K)
{
    return fp8BlockwiseCuteDslGemmSupported() && M >= 128 && N > 0 && K > 0 && N % 128 == 0 && K % 128 == 0;
}

cudaError_t launchFp8BlockwiseCuteDslGemm(void const* A, float const* SFA, void const* B, float const* SFB, void* D,
    int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    if (A == nullptr || SFA == nullptr || B == nullptr || SFB == nullptr || D == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (!fp8BlockwiseCuteDslGemmCanImplement(M, N, K))
    {
        return cudaErrorNotSupported;
    }
    if (!ensureModule(stream))
    {
        return cudaErrorUnknown;
    }

    fp8_blockwise_gemm_fp16_Tensor_a_t tensorA{};
    fp8_blockwise_gemm_fp16_Tensor_b_t tensorB{};
    fp8_blockwise_gemm_fp16_Tensor_c_t tensorC{};
    fp8_blockwise_gemm_fp16_Tensor_sfa_t tensorSfa{};
    fp8_blockwise_gemm_fp16_Tensor_sfb_t tensorSfb{};
    setTensor(tensorA, const_cast<void*>(A), M, K);
    setTensor(tensorB, const_cast<void*>(B), N, K);
    setTensor(tensorC, D, M, N);
    setTensor(tensorSfa, const_cast<float*>(SFA), M, K / 128);
    setTensor(tensorSfb, const_cast<float*>(SFB), N / 128, K / 128);

    int32_t const result = cute_dsl_fp8_blockwise_gemm_fp16_wrapper(
        &gModule.module, &tensorA, &tensorB, &tensorC, &tensorSfa, &tensorSfb, stream);
    if (result != 0)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

} // namespace trt_edgellm::kernel

#else

namespace trt_edgellm::kernel
{

bool fp8BlockwiseCuteDslGemmSupported()
{
    return false;
}

bool fp8BlockwiseCuteDslGemmCanImplement(int32_t, int32_t, int32_t)
{
    return false;
}

cudaError_t launchFp8BlockwiseCuteDslGemm(
    void const*, float const*, void const*, float const*, void*, int32_t, int32_t, int32_t, cudaStream_t)
{
    return cudaErrorNotSupported;
}

} // namespace trt_edgellm::kernel

#endif
