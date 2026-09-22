#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOT export the CuTe DSL FP8 blockwise GEMM used by FP8BlockGemmPlugin."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_runtime
import torch


def load_kernel_source(root: Path, directory: Path):
    """Load the vendored CuTe DSL example with the current fence API."""
    source = root / "3rdParty/cutlass/examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py"
    adapted_source = source.read_text()
    if not hasattr(cute.arch, "ProxyKind"):
        adapted_source = adapted_source.replace(
            "cute.arch.ProxyKind.async_shared", '"async.shared"')
    if not hasattr(cute.arch, "SharedSpace"):
        adapted_source = adapted_source.replace(
            "cute.arch.SharedSpace.shared_cta", '"cta"')
    module_path = directory / "blockwise_gemm.py"
    module_path.write_text(adapted_source)
    spec = importlib.util.spec_from_file_location("fp8_blockwise_gemm_cutedsl",
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load CuTe DSL source {source}")
    spec.loader.exec_module(module)
    return module


def make_dynamic_tensor(array):
    """Create a row-major three-dimensional dynamic CuTe DSL tensor."""
    tensor = cute_runtime.from_dlpack(array, assumed_align=16)
    return (tensor.mark_layout_dynamic(
        leading_dim=1).mark_compact_shape_dynamic(
            mode=0, stride_order=(2, 0, 1)).mark_compact_shape_dynamic(
                mode=1, stride_order=(2, 0, 1)))


def main() -> None:
    """Compile a shape-polymorphic 1-CTA FP8 blockwise GEMM and export its C ABI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--mnk", default="128,256,128")
    args = parser.parse_args()
    m, n, k = (int(value) for value in args.mnk.split(","))
    if min(m, n, k) <= 0 or n % 128 != 0 or k % 128 != 0:
        raise ValueError(
            "M/N/K must be positive and N/K must be multiples of 128")

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
            prefix="fp8_blockwise_gemm_cutedsl_") as directory:
        example = load_kernel_source(root, Path(directory))
        kernel = example.BlockwiseGemmKernel(cutlass.Float32, False,
                                             (128, 128), (1, 1))
        if not kernel.can_implement(cutlass.Float8E4M3FN, cutlass.Float32,
                                    cutlass.Float16, False, (128, 128),
                                    (1, 1), m, n, k, 1, "k", "k", "n"):
            raise ValueError(
                f"Unsupported CuTe DSL FP8 blockwise GEMM shape M={m}, N={n}, K={k}"
            )

        a = torch.zeros((m, k, 1), device="cuda", dtype=torch.float8_e4m3fn)
        b = torch.zeros((n, k, 1), device="cuda", dtype=torch.float8_e4m3fn)
        c = torch.zeros((m, n, 1), device="cuda", dtype=torch.float16)
        sfa = torch.ones((m, k // 128, 1), device="cuda", dtype=torch.float32)
        sfb = torch.ones((n // 128, k // 128, 1),
                         device="cuda",
                         dtype=torch.float32)
        tensors = tuple(
            make_dynamic_tensor(array) for array in (a, b, c, sfa, sfb))
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        max_active_clusters = cutlass.Int32(1)
        compiled = cute.compile(kernel, *tensors, max_active_clusters, stream)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        compiled.export_to_c(file_path=args.output_dir,
                             file_name=args.file_name,
                             function_prefix=args.function_prefix)


if __name__ == "__main__":
    main()
