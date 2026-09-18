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

//! FP8 per-token, per-128-element-group dynamic activation quantization.
//!
//! Mirrors the Qwen3 / DeepSeek "per_token_group_quant_fp8" recipe used with
//! 128x128 block-wise FP8 weights: for each token row and each contiguous group
//! of ``kFp8QuantGroupSize`` elements along K, compute amax, derive
//! ``scale = amax / 448`` (E4M3 finite max), and store ``q = x / scale`` as
//! FP8 E4M3. The scale layout is K-major ``[M, K / group]`` row-major so it
//! feeds the CUTLASS blockwise GEMM ``SFA`` operand with no repack.
constexpr int32_t kFp8QuantGroupSize = 128;

//! @param input   FP16 activations, row-major ``[M, K]`` (K contiguous).
//! @param outputQ FP8 E4M3 output, row-major ``[M, K]`` (bit-reinterpreted bytes).
//! @param outputScale FP32 scales, row-major ``[M, K / kFp8QuantGroupSize]``.
//! @param M       Number of tokens (rows).
//! @param K       Reduction dim; must be a multiple of kFp8QuantGroupSize.
//! @param stream  CUDA stream.
void launchFp8PerTokenGroupQuantize(
    void const* input, void* outputQ, void* outputScale, int32_t M, int32_t K, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
