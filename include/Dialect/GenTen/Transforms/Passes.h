//===-- Passes.h - GEN_TEN optimization pass declarations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the optimization passes for the GEN_TEN Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gen_ten {

std::unique_ptr<Pass> createGenTenCleanUpPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/GenTen/Transforms/Passes.h.inc"

} // namespace gen_ten
} // namespace mlir

#endif // MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES_H
