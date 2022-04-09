//===- GenTenOps.cpp - MLIR Dialect for GEN_TEN --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the GEN_TEN Specification:
// https://some_url/
//
//===----------------------------------------------------------------------===//

#include "Dialect/GenTen/IR/GenTenOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::gen_ten;

#include "Dialect/GenTen/IR/GenTenOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// GenTen dialect structs and interface includes.
//===----------------------------------------------------------------------===//
#include "Dialect/GenTen/IR/GenTenInterfaces.cpp.inc"
#include "Dialect/GenTen/IR/GenTenStructs.cpp.inc"

namespace {
////===----------------------------------------------------------------------===//
//// Dialect Function Inliner Interface.
////===----------------------------------------------------------------------===//
//struct GenTenInlinerInterface : public DialectInlinerInterface {
//  using DialectInlinerInterface::DialectInlinerInterface;
//
//  //===--------------------------------------------------------------------===//
//  // Analysis Hooks.
//  //===--------------------------------------------------------------------===//
//
//  /// All operations can be inlined by default.
//  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
//                       BlockAndValueMapping &map) const final {
//    return true;
//  }
//
//  /// All regions with If and While parent operators can be inlined.
//  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
//                       BlockAndValueMapping &map) const final {
//    return false;
//  }
//};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// GenTen dialect initialization.
//===----------------------------------------------------------------------===//

void GenTenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/GenTen/IR/GenTenOps.cpp.inc"
      >();
//  addInterfaces<GenTenInlinerInterface>();
}

Operation *GenTenDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // GenTen dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
//  if (value.isa<ElementsAttr>())
//    return builder.create<mlir::gen_ten::ConstantOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// FunctionOp
//===----------------------------------------------------------------------===//

FunctionOp FunctionOp::create(Location location, StringRef name, FunctionType type,
                        ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FunctionOp::build(builder, state, name, type, attrs);
  return cast<FunctionOp>(Operation::create(state));
}

void FunctionOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  
  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
}

//===----------------------------------------------------------------------===//
// GEN_TEN Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/GenTen/IR/GenTenOps.cpp.inc"

