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

#include "fp8BlockwiseGemmRunner.h"

#include <stdexcept>
#include <string>

// The whole CUTLASS blockwise path is compiled only when CMake detects a
// Blackwell datacenter target arch (tcgen05). Otherwise this TU provides
// inert stubs so the plugin still links and fails cleanly at runtime.
#if defined(EDGELLM_FP8_BLOCKWISE_GEMM_ENABLED)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

using namespace cute;

// ---- Kernel configuration (verified on Thor / SM110 in Phase 0) -------------
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor; // activations [M,K], K contiguous
constexpr int kAlignA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor; // weight [N,K] row-major == B[K,N] col-major
constexpr int kAlignB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementD = cutlass::half_t;
using LayoutD = cutlass::layout::RowMajor; // output [M,N], N contiguous
constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute = float;

// 1-SM MMA, 1x1x1 cluster: SM110 (Thor) has no 2-SM UMMA.
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// SFA granularity (1,128) per-token-per-K-block; SFB granularity (128,128).
// K-major for both -> SFA [M,K/128] row-major, SFB [N/128,K/128] row-major.
using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, UMMA::Major::K, UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute, void, LayoutD, kAlignD, ElementD, LayoutD, kAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA>, kAlignA, ElementB,
    cute::tuple<LayoutB, LayoutSFB>, kAlignB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using GemmKernel
    = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;

constexpr int32_t k2CtaM{4096};
constexpr int32_t k2CtaN{2560};
constexpr int32_t k2CtaK{9728};

using MmaTileShape2Cta_MNK = Shape<_256, _256, _128>;
using ClusterShape2Cta_MNK = Shape<_2, _1, _1>;
using ScaleConfig2Cta = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, UMMA::Major::MN, UMMA::Major::MN>;
using LayoutSFA2Cta = decltype(ScaleConfig2Cta::deduce_layoutSFA());
using LayoutSFB2Cta = decltype(ScaleConfig2Cta::deduce_layoutSFB());

using CollectiveEpilogue2Cta = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp, MmaTileShape2Cta_MNK, ClusterShape2Cta_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, void, LayoutD, kAlignD,
    ElementD, LayoutD, kAlignD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop2Cta = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp, ElementA, cute::tuple<LayoutA, LayoutSFA2Cta>, kAlignA, ElementB,
    cute::tuple<LayoutB, LayoutSFB2Cta>, kAlignB, ElementAccumulator, MmaTileShape2Cta_MNK, ClusterShape2Cta_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue2Cta::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>::CollectiveOp;

using GemmKernel2Cta = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop2Cta,
    CollectiveEpilogue2Cta, void>;
using Gemm2Cta = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2Cta>;

using StrideA2Cta = typename Gemm2Cta::GemmKernel::StrideA;
using StrideB2Cta = typename Gemm2Cta::GemmKernel::StrideB;
using StrideD2Cta = typename Gemm2Cta::GemmKernel::StrideD;

// Blackwell datacenter archs with tcgen05: SM100/101/103 (major 10) and
// SM110 = Thor (major 11). SM120/121 (major 12, GeForce Blackwell) lack
// tcgen05 and are excluded.
bool deviceIsBlackwellDatacenter()
{
    static int cached = -1;
    if (cached >= 0)
    {
        return cached == 1;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        cached = 0;
        return false;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess)
    {
        cached = 0;
        return false;
    }
    cached = (props.major == 10 || props.major == 11) ? 1 : 0;
    return cached == 1;
}

typename Gemm::Arguments makeArguments(
    void const* A, void const* SFA, void const* B, void const* SFB, void* D, int32_t M, int32_t N, int32_t K)
{
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {static_cast<ElementA const*>(A), stride_A, static_cast<ElementB const*>(B), stride_B,
            static_cast<ElementAccumulator const*>(SFA), layout_SFA, static_cast<ElementAccumulator const*>(SFB),
            layout_SFB},
        {{}, nullptr, stride_D, static_cast<ElementD*>(D), stride_D}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    return args;
}

bool is2CtaShape(int32_t M, int32_t N, int32_t K)
{
    return M == k2CtaM && N == k2CtaN && K == k2CtaK;
}

typename Gemm2Cta::Arguments make2CtaArguments(void const* A, void const* SFA, void const* B, void const* SFB, void* D)
{
    auto stride_A = cutlass::make_cute_packed_stride(StrideA2Cta{}, {k2CtaM, k2CtaK, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB2Cta{}, {k2CtaN, k2CtaK, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD2Cta{}, {k2CtaM, k2CtaN, 1});
    auto layout_SFA = ScaleConfig2Cta::tile_atom_to_shape_SFA(make_shape(k2CtaM, k2CtaN, k2CtaK, 1));
    auto layout_SFB = ScaleConfig2Cta::tile_atom_to_shape_SFB(make_shape(k2CtaM, k2CtaN, k2CtaK, 1));

    typename Gemm2Cta::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {k2CtaM, k2CtaN, k2CtaK, 1},
        {static_cast<ElementA const*>(A), stride_A, static_cast<ElementB const*>(B), stride_B,
            static_cast<ElementAccumulator const*>(SFA), layout_SFA, static_cast<ElementAccumulator const*>(SFB),
            layout_SFB},
        {{}, nullptr, stride_D, static_cast<ElementD*>(D), stride_D}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    return args;
}

} // namespace

bool fp8BlockwiseGemmSupported()
{
    return deviceIsBlackwellDatacenter();
}

size_t fp8BlockwiseGemmWorkspaceSize(int32_t M, int32_t N, int32_t K)
{
    if (!deviceIsBlackwellDatacenter() || M <= 0 || N <= 0 || K <= 0)
    {
        return 0;
    }
    auto args = makeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K);
    return static_cast<size_t>(Gemm::get_workspace_size(args));
}

bool fp8BlockwiseGemm2CtaSupported(int32_t M, int32_t N, int32_t K)
{
    return deviceIsBlackwellDatacenter() && is2CtaShape(M, N, K);
}

size_t fp8BlockwiseGemm2CtaWorkspaceSize(int32_t M, int32_t N, int32_t K)
{
    if (!fp8BlockwiseGemm2CtaSupported(M, N, K))
    {
        return 0;
    }
    auto args = make2CtaArguments(nullptr, nullptr, nullptr, nullptr, nullptr);
    return static_cast<size_t>(Gemm2Cta::get_workspace_size(args));
}

void launchFp8BlockwiseGemm(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int32_t M,
    int32_t N, int32_t K, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    if (!deviceIsBlackwellDatacenter())
    {
        throw std::runtime_error("FP8 blockwise GEMM requires a Blackwell datacenter GPU (SM100/SM110).");
    }
    if (M <= 0 || N <= 0 || K <= 0 || (K % 128) != 0 || (N % 128) != 0)
    {
        throw std::runtime_error("FP8 blockwise GEMM requires N,K > 0 and multiples of 128 (got M=" + std::to_string(M)
            + ", N=" + std::to_string(N) + ", K=" + std::to_string(K) + ").");
    }

    auto args = makeArguments(A, SFA, B, SFB, D, M, N, K);
    Gemm gemm;
    size_t const required = static_cast<size_t>(Gemm::get_workspace_size(args));
    if (required > workspaceSize)
    {
        throw std::runtime_error(
            "FP8 blockwise GEMM workspace too small: need " + std::to_string(required) + " bytes.");
    }
    cutlass::Status status = gemm.run(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        throw std::runtime_error(std::string("FP8 blockwise GEMM failed: ") + cutlassGetStatusString(status));
    }
}

void launchFp8BlockwiseGemm2Cta(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int32_t M,
    int32_t N, int32_t K, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    if (!fp8BlockwiseGemm2CtaSupported(M, N, K))
    {
        throw std::runtime_error("FP8 blockwise 2-CTA GEMM only supports M=4096, N=2560, K=9728.");
    }

    auto args = make2CtaArguments(A, SFA, B, SFB, D);
    Gemm2Cta gemm;
    size_t const required = static_cast<size_t>(Gemm2Cta::get_workspace_size(args));
    if (required > workspaceSize)
    {
        throw std::runtime_error(
            "FP8 blockwise 2-CTA GEMM workspace too small: need " + std::to_string(required) + " bytes.");
    }
    cutlass::Status status = gemm.run(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        throw std::runtime_error(std::string("FP8 blockwise 2-CTA GEMM failed: ") + cutlassGetStatusString(status));
    }
}

} // namespace kernel
} // namespace trt_edgellm

#else // !EDGELLM_FP8_BLOCKWISE_GEMM_ENABLED

namespace trt_edgellm
{
namespace kernel
{

bool fp8BlockwiseGemmSupported()
{
    return false;
}

size_t fp8BlockwiseGemmWorkspaceSize(int32_t, int32_t, int32_t)
{
    return 0;
}

bool fp8BlockwiseGemm2CtaSupported(int32_t, int32_t, int32_t)
{
    return false;
}

size_t fp8BlockwiseGemm2CtaWorkspaceSize(int32_t, int32_t, int32_t)
{
    return 0;
}

void launchFp8BlockwiseGemm(
    void const*, void const*, void const*, void const*, void*, int32_t, int32_t, int32_t, void*, size_t, cudaStream_t)
{
    throw std::runtime_error(
        "FP8 blockwise GEMM kernel was not compiled for this target arch (needs Blackwell datacenter SM100/SM110).");
}

void launchFp8BlockwiseGemm2Cta(
    void const*, void const*, void const*, void const*, void*, int32_t, int32_t, int32_t, void*, size_t, cudaStream_t)
{
    throw std::runtime_error(
        "FP8 blockwise 2-CTA GEMM kernel was not compiled for this target arch (needs Blackwell datacenter "
        "SM100/SM110).");
}

} // namespace kernel
} // namespace trt_edgellm

#endif // EDGELLM_FP8_BLOCKWISE_GEMM_ENABLED
