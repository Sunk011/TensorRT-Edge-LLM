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

#include <cstdint>

namespace trt_edgellm::kernel
{

//! True when the CuTe DSL FP8 blockwise kernel is compiled for this device.
bool fp8BlockwiseCuteDslGemmSupported();

//! True when the CuTe DSL kernel supports this dynamic GEMM shape.
bool fp8BlockwiseCuteDslGemmCanImplement(int32_t M, int32_t N, int32_t K);

//! Launch FP8 E4M3 GEMM with FP32 1x128 activation and 128x128 weight scales.
cudaError_t launchFp8BlockwiseCuteDslGemm(void const* A, float const* SFA, void const* B, float const* SFB, void* D,
    int32_t M, int32_t N, int32_t K, cudaStream_t stream);

} // namespace trt_edgellm::kernel
