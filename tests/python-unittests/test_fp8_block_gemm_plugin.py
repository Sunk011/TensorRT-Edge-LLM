# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import pytest
import torch

from tensorrt_edgellm.models.linear import (FP8BlockLinear,
                                            prepare_fp8_block_weights)
from tensorrt_edgellm.models.ops import fp8_block_gemm_backend_id
from tensorrt_edgellm.onnx.dynamo_translations import \
    build_custom_translation_table


def test_fp8_block_gemm_backend_is_serialized(tmp_path):
    linear = FP8BlockLinear(128, 128, backend=1).eval()
    linear.weight.copy_(torch.randn(128, 128).to(torch.float8_e4m3fn))
    prepare_fp8_block_weights(linear)

    output = tmp_path / "model.onnx"
    torch.onnx.export(
        linear,
        (torch.randn(1, 2, 128, dtype=torch.float16), ),
        output,
        dynamo=True,
        custom_translation_table=build_custom_translation_table(),
        opset_version=21,
    )

    nodes = [
        node for node in onnx.load(output).graph.node
        if node.op_type == "FP8BlockGemmPlugin"
    ]
    assert len(nodes) == 1
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in nodes[0].attribute
    }
    assert attributes == {"backend": 1, "gemm_k": 128, "gemm_n": 128}


@pytest.mark.parametrize("backend, expected", [("cutlass", 0), ("cute_dsl", 1),
                                               ("auto", 2)])
def test_fp8_block_gemm_backend_ids(backend, expected):
    assert fp8_block_gemm_backend_id(backend) == expected


def test_fp8_block_gemm_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown FP8 block GEMM backend"):
        fp8_block_gemm_backend_id("not-a-backend")
