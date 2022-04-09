//===- PassDetail.h - GEN_TEN Pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSDETAIL_H
#define MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "Dialect/GenTen/IR/GenTenOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "Dialect/GenTen/Transforms/Passes.h.inc"

} // end namespace mlir

#endif // MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSDETAIL_H

