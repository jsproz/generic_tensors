//===- GenTenCompilerGenTenGen.h - MLIR Generation from a GenTen Binary AST -----------*- C++ -*------===//
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation
// from a Module AST for the GenTen Text language.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_GEN_TEN_GEN_H
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_GEN_TEN_GEN_H

#include <memory>

namespace gen_ten_compiler {

struct ProgramAST;
struct ProgramIR;

std::shared_ptr<gen_ten_compiler::ProgramIR> genTenGen( std::shared_ptr<gen_ten_compiler::ProgramAST> program );

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_GEN_TEN_GEN_H
