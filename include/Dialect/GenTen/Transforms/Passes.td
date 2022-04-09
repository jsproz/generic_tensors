//===-- Passes.td - GEN_TEN optimization pass declarations ----*- tablegen -*-===//
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

#ifndef MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES
#define MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def GenTenCleanUp : Pass<"gen_ten-clean-up","mlir::gen_ten::FunctionOp"> {
  let summary = "Clean up GEN_TEN after construction";
  let description = [{
    Pass that removes unneeded operations in GEN_TEN after construction.
  }];

  let constructor = "createGenTenCleanUpPass()";
  let dependentDialects = [
    "gen_ten::GenTenDialect",
  ];
}

#endif // MLIR_DIALECT_GEN_TEN_TRANSFORMS_PASSES
