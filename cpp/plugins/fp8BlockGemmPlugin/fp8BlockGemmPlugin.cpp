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

#include "fp8BlockGemmPlugin.h"

#include "common/logger.h"
#include "kernels/fp8BlockwiseGemm/cuteDslFp8BlockwiseGemmRunner.h"
#include "kernels/fp8BlockwiseGemm/fp8BlockwiseGemmRunner.h"
#include "kernels/fp8BlockwiseGemm/fp8BlockwiseScaleTranspose.h"
#include "kernels/fp8BlockwiseGemm/fp8PerTokenGroupQuantize.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kPLUGIN_NAME{"FP8BlockGemmPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kBLOCK_SIZE{128};
constexpr int32_t kBackendCutlass{0};
constexpr int32_t kBackendCuteDsl{1};
constexpr int32_t kBackendAuto{2};

// 256-byte alignment keeps the FP8 activation buffer, the FP32 SFA buffer and
// the CUTLASS workspace independently aligned inside one TRT workspace blob.
constexpr size_t kAlign{256};
inline size_t alignUp(size_t x, size_t a)
{
    return (x + a - 1) & ~(a - 1);
}

bool isValidBackend(int32_t backend)
{
    return backend == kBackendCutlass || backend == kBackendCuteDsl || backend == kBackendAuto;
}
} // namespace

PluginFieldCollection FP8BlockGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> FP8BlockGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FP8BlockGemmPluginCreator);

FP8BlockGemmPlugin::FP8BlockGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t backend)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mBackend(backend)
{
}

FP8BlockGemmPlugin::FP8BlockGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "backend")
        {
            mBackend = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
    if (!isValidBackend(mBackend))
    {
        throw std::invalid_argument("FP8BlockGemmPlugin: invalid backend");
    }
}

IPluginCapability* FP8BlockGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::getCapabilityInterface failed: %s", e.what());
        return nullptr;
    }
}

IPluginV3* FP8BlockGemmPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new FP8BlockGemmPlugin(mLayerName, mGemmN, mGemmK, mBackend);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::clone failed: %s", e.what());
        return nullptr;
    }
}

char const* FP8BlockGemmPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* FP8BlockGemmPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* FP8BlockGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void FP8BlockGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

int32_t FP8BlockGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t FP8BlockGemmPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    try
    {
        outputTypes[0] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::getOutputDataTypes failed: %s", e.what());
        return -1;
    }
}

int32_t FP8BlockGemmPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/, DimsExprs* outputs, int32_t /*nbOutputs*/,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::getOutputShapes failed: %s", e.what());
        return -1;
    }
}

bool FP8BlockGemmPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        auto const& desc = inOut[pos].desc;
        bool ok = desc.format == PluginFormat::kLINEAR;
        switch (pos)
        {
        case 0: // activation: fp16 [b, seq, K]
            ok &= desc.type == DataType::kHALF;
            ok &= desc.dims.nbDims == 3;
            ok &= desc.dims.d[2] == mGemmK;
            break;
        case 1: // weight: int8 (fp8 bits) [N, K]
            ok &= desc.type == DataType::kINT8;
            ok &= desc.dims.nbDims == 2;
            ok &= desc.dims.d[0] == mGemmN;
            ok &= desc.dims.d[1] == mGemmK;
            break;
        case 2: // weight scale: fp32 [N/128, K/128]
            ok &= desc.type == DataType::kFLOAT;
            ok &= desc.dims.nbDims == 2;
            ok &= desc.dims.d[0] == mGemmN / kBLOCK_SIZE;
            ok &= desc.dims.d[1] == mGemmK / kBLOCK_SIZE;
            break;
        case 3: // output: fp16 [b, seq, N]
            ok &= desc.type == DataType::kHALF;
            ok &= desc.dims.nbDims == 3;
            ok &= desc.dims.d[2] == mGemmN;
            break;
        default: ok = false; break;
        }
        return ok;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::supportsFormatCombination failed: %s", e.what());
        return false;
    }
}

int32_t FP8BlockGemmPlugin::configurePlugin(DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

size_t FP8BlockGemmPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    try
    {
        if (nbInputs < 1 || inputs[0].max.nbDims < 2 || mGemmK <= 0)
        {
            return 0;
        }
        int64_t const mMax = static_cast<int64_t>(inputs[0].max.d[0]) * inputs[0].max.d[1];
        if (mMax <= 0)
        {
            return 0;
        }
        int64_t const kb = mGemmK / kBLOCK_SIZE;
        size_t const aBytes = alignUp(static_cast<size_t>(mMax) * static_cast<size_t>(mGemmK), kAlign);
        size_t const sfaBytes = alignUp(static_cast<size_t>(mMax) * static_cast<size_t>(kb) * sizeof(float), kAlign);
        bool const use2Cta = kernel::fp8BlockwiseGemm2CtaSupported(static_cast<int32_t>(mMax), mGemmN, mGemmK);
        if (!use2Cta)
        {
            size_t const cutlassBytes
                = kernel::fp8BlockwiseGemmWorkspaceSize(static_cast<int32_t>(mMax), mGemmN, mGemmK);
            return aBytes + sfaBytes + cutlassBytes;
        }
        int64_t const nb = mGemmN / kBLOCK_SIZE;
        size_t const sfbBytes = alignUp(static_cast<size_t>(nb) * static_cast<size_t>(kb) * sizeof(float), kAlign);
        size_t const cutlassBytes
            = kernel::fp8BlockwiseGemm2CtaWorkspaceSize(static_cast<int32_t>(mMax), mGemmN, mGemmK);
        return aBytes + sfaBytes + sfaBytes + sfbBytes + cutlassBytes;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::getWorkspaceSize failed: %s", e.what());
        return 0;
    }
}

int32_t FP8BlockGemmPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int64_t const m64 = static_cast<int64_t>(inputDesc[0].dims.d[0]) * inputDesc[0].dims.d[1];
        if (m64 <= 0)
        {
            return 0;
        }
        int32_t const M = static_cast<int32_t>(m64);
        int32_t const K = mGemmK;
        int32_t const N = mGemmN;
        int64_t const kb = K / kBLOCK_SIZE;

        auto* base = static_cast<uint8_t*>(workspace);
        uint8_t* aFp8 = base;
        size_t const aBytes = alignUp(static_cast<size_t>(M) * static_cast<size_t>(K), kAlign);
        float* sfa = reinterpret_cast<float*>(base + aBytes);
        size_t const sfaBytes = alignUp(static_cast<size_t>(M) * static_cast<size_t>(kb) * sizeof(float), kAlign);

        // 1) Dynamic per-token, per-128-K-group activation quantization (fp16 -> fp8 + SFA).
        kernel::launchFp8PerTokenGroupQuantize(inputs[0], aFp8, sfa, M, K, stream);

        bool const cuteDslCanImplement = kernel::fp8BlockwiseCuteDslGemmCanImplement(M, N, K);
        bool const useCuteDsl = (mBackend == kBackendCuteDsl && cuteDslCanImplement)
            || (mBackend == kBackendAuto && cuteDslCanImplement && M >= kBLOCK_SIZE);
        if (useCuteDsl)
        {
            cudaError_t const error = kernel::launchFp8BlockwiseCuteDslGemm(
                aFp8, sfa, inputs[1], static_cast<float const*>(inputs[2]), outputs[0], M, N, K, stream);
            if (error != cudaSuccess)
            {
                LOG_ERROR("FP8BlockGemmPlugin: CuTe DSL FP8 blockwise GEMM failed: %s", cudaGetErrorString(error));
                return -1;
            }
            return 0;
        }

        if (!kernel::fp8BlockwiseGemmSupported())
        {
            LOG_ERROR("FP8BlockGemmPlugin: no FP8 blockwise backend supports M=%d, N=%d, K=%d.", M, N, K);
            return -1;
        }

        if (kernel::fp8BlockwiseGemm2CtaSupported(M, N, K))
        {
            int64_t const nb = N / kBLOCK_SIZE;
            auto* sfaMn = reinterpret_cast<float*>(base + aBytes + sfaBytes);
            size_t const sfbBytes = alignUp(static_cast<size_t>(nb) * static_cast<size_t>(kb) * sizeof(float), kAlign);
            auto* sfbMn = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sfaMn) + sfaBytes);
            void* cutlassWs = static_cast<void*>(reinterpret_cast<uint8_t*>(sfbMn) + sfbBytes);
            size_t const cutlassWsSize = kernel::fp8BlockwiseGemm2CtaWorkspaceSize(M, N, K);

            kernel::launchFp8BlockwiseScaleTranspose(sfa, sfaMn, M, static_cast<int32_t>(kb), stream);
            kernel::launchFp8BlockwiseScaleTranspose(static_cast<float const*>(inputs[2]), sfbMn,
                static_cast<int32_t>(nb), static_cast<int32_t>(kb), stream);
            kernel::launchFp8BlockwiseGemm2Cta(
                aFp8, sfaMn, inputs[1], sfbMn, outputs[0], M, N, K, cutlassWs, cutlassWsSize, stream);
        }
        else
        {
            void* cutlassWs = static_cast<void*>(base + aBytes + sfaBytes);
            size_t const cutlassWsSize = kernel::fp8BlockwiseGemmWorkspaceSize(M, N, K);
            kernel::launchFp8BlockwiseGemm(
                aFp8, sfa, inputs[1], inputs[2], outputs[0], M, N, K, cutlassWs, cutlassWsSize, stream);
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPlugin::enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t FP8BlockGemmPlugin::onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

IPluginV3* FP8BlockGemmPlugin::attachToContext(IPluginResourceContext* /*context*/) noexcept
{
    return clone();
}

PluginFieldCollection const* FP8BlockGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("backend", &mBackend, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

FP8BlockGemmPluginCreator::FP8BlockGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backend", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* FP8BlockGemmPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* FP8BlockGemmPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* FP8BlockGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* FP8BlockGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void FP8BlockGemmPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

IPluginV3* FP8BlockGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /*phase*/) noexcept
{
    try
    {
        auto* plugin = new FP8BlockGemmPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FP8BlockGemmPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
