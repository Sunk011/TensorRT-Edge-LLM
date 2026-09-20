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

#include "fp8BlockwiseScaleTranspose.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kTileSize{16};

__global__ void fp8BlockwiseScaleTransposeKernel(float const* input, float* output, int32_t rows, int32_t columns)
{
    __shared__ float tile[kTileSize][kTileSize + 1];

    int32_t const inputColumn = static_cast<int32_t>(blockIdx.x) * kTileSize + threadIdx.x;
    int32_t const inputRow = static_cast<int32_t>(blockIdx.y) * kTileSize + threadIdx.y;
    if (inputRow < rows && inputColumn < columns)
    {
        tile[threadIdx.y][threadIdx.x] = input[static_cast<size_t>(inputRow) * columns + inputColumn];
    }
    __syncthreads();

    int32_t const outputColumn = static_cast<int32_t>(blockIdx.y) * kTileSize + threadIdx.x;
    int32_t const outputRow = static_cast<int32_t>(blockIdx.x) * kTileSize + threadIdx.y;
    if (outputRow < columns && outputColumn < rows)
    {
        output[static_cast<size_t>(outputRow) * rows + outputColumn] = tile[threadIdx.x][threadIdx.y];
    }
}

} // namespace

void launchFp8BlockwiseScaleTranspose(
    float const* input, float* output, int32_t rows, int32_t columns, cudaStream_t stream)
{
    if (rows <= 0 || columns <= 0)
    {
        return;
    }
    dim3 const block{kTileSize, kTileSize};
    dim3 const grid{static_cast<uint32_t>((columns + kTileSize - 1) / kTileSize),
        static_cast<uint32_t>((rows + kTileSize - 1) / kTileSize)};
    fp8BlockwiseScaleTransposeKernel<<<grid, block, 0, stream>>>(input, output, rows, columns);
}

} // namespace kernel
} // namespace trt_edgellm
