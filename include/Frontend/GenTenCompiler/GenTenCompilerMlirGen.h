//===- GenTenCompilerMlirGen.h - MLIR Generation from a GenTen Binary AST -----------*- C++ -*------===//
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation
// from a Module AST for the GenTen Binary language.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_MLIR_GEN_H
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_MLIR_GEN_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class MLIRContext;
template<class T>
class OwningOpRef;
} // namespace mlir

namespace gen_ten_compiler {
struct ProgramIR;
} // namespace gen_ten_compiler

namespace gen_ten_compiler {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen( mlir::MLIRContext &context, std::shared_ptr<gen_ten_compiler::ProgramIR> programIR );

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_MLIR_GEN_H
