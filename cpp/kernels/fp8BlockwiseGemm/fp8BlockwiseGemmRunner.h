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

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! Block-wise FP8 GEMM (Qwen3 / DeepSeek recipe) via the CUTLASS SM100
//! blockwise-scaled collective, compiled for Blackwell datacenter archs.
//!
//! Computes ``D[M,N] (fp16) = dequant(A) @ dequant(B)^T`` where:
//!   - A is FP8 E4M3 ``[M,K]`` row-major with per-token, per-128-K scales
//!     ``SFA[M, K/128]`` (row-major / K-major),
//!   - B is the FP8 E4M3 weight ``[N,K]`` row-major with per-128x128-block
//!     scales ``SFB[N/128, K/128]`` (row-major / K-major) == the checkpoint's
//!     ``weight_scale_inv`` verbatim.
//!
//! The CUTLASS scale config is ``Sm100BlockwiseScaleConfig<1,128,128,K,K>`` so
//! neither operand needs a transpose. K must be a multiple of 128; N a
//! multiple of 128 (Qwen3-VL satisfies both).

//! @return true iff the active device is Blackwell datacenter (SM 10.x/11.x)
//! and this build compiled the CUTLASS blockwise kernel.
bool fp8BlockwiseGemmSupported();

//! @return workspace bytes required by launchFp8BlockwiseGemm for this shape
//! (currently 0 for the 1x1x1-cluster configuration, but queried dynamically).
size_t fp8BlockwiseGemmWorkspaceSize(int32_t M, int32_t N, int32_t K);

//! @return true iff the fixed 4096x2560x9728 2-CTA specialization is usable.
bool fp8BlockwiseGemm2CtaSupported(int32_t M, int32_t N, int32_t K);

//! @return workspace bytes required by the fixed 2-CTA specialization.
size_t fp8BlockwiseGemm2CtaWorkspaceSize(int32_t M, int32_t N, int32_t K);

//! Launch the block-wise FP8 GEMM. Throws std::runtime_error on misuse
//! (unsupported device, invalid shape) or CUTLASS failure.
//! @param A     FP8 E4M3 activations ``[M,K]`` (bytes).
//! @param SFA   FP32 activation scales ``[M, K/128]`` row-major.
//! @param B     FP8 E4M3 weights ``[N,K]`` (bytes).
//! @param SFB   FP32 weight scales ``[N/128, K/128]`` row-major.
//! @param D     FP16 output ``[M,N]`` row-major.
void launchFp8BlockwiseGemm(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int32_t M,
    int32_t N, int32_t K, void* workspace, size_t workspaceSize, cudaStream_t stream);

//! Launches the fixed 2-CTA specialization. SFA and SFB use MN-major layouts:
//! ``SFA[K/128, M]`` and ``SFB[K/128, N/128]``.
void launchFp8BlockwiseGemm2Cta(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int32_t M,
    int32_t N, int32_t K, void* workspace, size_t workspaceSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
