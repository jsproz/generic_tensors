//===-- GenTenOps.h - GEN_TEN dialect operation definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the GEN_TEN Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_OPS_H
#define MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// GEN_TEN dialect and structs includes.
//===----------------------------------------------------------------------===//
#include "Dialect/GenTen/IR/GenTenOpsDialect.h.inc"
#include "Dialect/GenTen/IR/GenTenStructs.h.inc"

namespace mlir {
namespace gen_ten {

#include "Dialect/GenTen/IR/GenTenInterfaces.h.inc"

} // end namespace gen_ten
} // end namespace mlir

#define GET_OP_CLASSES
#include "Dialect/GenTen/IR/GenTenOps.h.inc"

#endif // MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_OPS_H
