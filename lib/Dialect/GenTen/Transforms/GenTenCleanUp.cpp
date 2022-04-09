//===- GenTenCleanUp.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remove undesirable and trivially optimizable GEN_TEN operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "Dialect/GenTen/IR/GenTenOps.h"
#include "Dialect/GenTen/Transforms/PassDetail.h"
#include "Dialect/GenTen/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::gen_ten;

namespace {

void cleanUpInRegion(Region &region) {

  for (auto &block : region) {
    SmallVector<Operation*, 4> toRemove;
    for (Operation &op : block) {
      if (op.getDialect()->getNamespace() != gen_ten::GenTenDialect::getDialectNamespace())
        continue;

      // make all identity ops dead by making the result use the input instead
      if ( isa<IdentityOp>( op ) ) {
        op.getResult( 0 ).replaceAllUsesWith( op.getOperand( 0 ) );
        toRemove.push_back( &op );
      }
    }

    // remove all the undesirable operations after scanning the block.
    for (auto op : toRemove) {
        op->erase();        
    }
  }
}

 
/// Pass that performs clean up in GEN_TEN after construction.
struct GenTenCleanUp : public GenTenCleanUpBase<GenTenCleanUp> {
public:
  void runOnOperation() override {

    FunctionOp theFunction = getOperation();

    // for all graphs in the model, clean up their region
    cleanUpInRegion( theFunction.getRegion() );

  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::gen_ten::createGenTenCleanUpPass() {
  return std::make_unique<GenTenCleanUp>();
}

