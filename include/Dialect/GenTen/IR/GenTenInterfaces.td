//===-- GenTenInterfaces.td - GEN_TEN dialect interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dialect op interfaces for the GEN_TEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_OP_INTERFACES
#define GEN_TEN_OP_INTERFACES

include "mlir/IR/OpBase.td"

def GenTenOpInterface : OpInterface<"GenTenOp"> {
  let description = [{
    Implemented by ops that correspond to the GenTen specification.
  }];
}

#endif
