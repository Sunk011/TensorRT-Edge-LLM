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

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
/*!
 * @brief TensorRT plugin for Qwen3 / DeepSeek-style block-wise FP8 GEMM.
 *
 * Runs a true FP8 (E4M3) tensor-core GEMM with 2-D block scaling on Blackwell
 * datacenter GPUs (SM100/SM110). The activation is dynamically quantized inside
 * enqueue (per-token, per-128-K group) then fed to the CUTLASS SM100 blockwise
 * collective together with the checkpoint's 128x128 block weight scales.
 *
 * Inputs:
 *   [0] x          fp16  [b, seq, K]   activation (M = b*seq tokens)
 *   [1] weight     int8  [N, K]        FP8 E4M3 weight bytes (bit-reinterpreted)
 *   [2] wscale     fp32  [N/128, K/128] per-block weight scale (weight_scale_inv)
 * Output:
 *   [0] y          fp16  [b, seq, N]
 */
class FP8BlockGemmPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    FP8BlockGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t backend);
    FP8BlockGemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    FP8BlockGemmPlugin() = delete;
    FP8BlockGemmPlugin(FP8BlockGemmPlugin const&) = delete;
    ~FP8BlockGemmPlugin() override = default;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mGemmN{};
    int32_t mGemmK{};
    int32_t mBackend{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/*!
 * @brief Factory for creating FP8BlockGemmPlugin instances.
 */
class FP8BlockGemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    FP8BlockGemmPluginCreator();
    ~FP8BlockGemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
