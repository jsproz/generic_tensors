//===- GenTenCompilerParser.h - Parser for the GenTen language -*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file defines a Parser interface for the GenTen language.
//
// This file was generated.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_PARSER_H
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_PARSER_H

#include "Frontend/GenTenCompiler/GenTenCompilerAST.hpp"
#include "Frontend/GenTenCompiler/GenTenCompilerLexer.hpp"

namespace gen_ten_compiler {

class Parser {
public:
  Parser( const TokenList &tokenList );
  
  std::shared_ptr<ProgramAST> parseProgram();
private:
    const TokenList &tokenList;
};

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_PARSER_H
