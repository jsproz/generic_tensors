//===- gen_ten_compiler.cpp - The GenTen Compiler ----------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the GenTen text compiler.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include "Dialect/GenTen/IR/GenTenCompilerIR.hpp"
#include "Dialect/GenTen/IR/GenTenPrinter.h"
#include "Dialect/GenTen/IR/GenTenOps.h"
#include "Dialect/GenTen/Transforms/Passes.h"
#include "Frontend/GenTenCompiler/GenTenCompilerAST.hpp"
#include "Frontend/GenTenCompiler/GenTenCompilerGenTenGen.h"
#include "Frontend/GenTenCompiler/GenTenCompilerLexer.hpp"
#include "Frontend/GenTenCompiler/GenTenCompilerParser.hpp"
#include "Frontend/GenTenCompiler/GenTenLogger.h"
#include "Frontend/GenTenCompiler/GenTenCompilerMlirGen.h"

void printUsage()
{
  std::cout << "Usage" << std::endl;
  std::cout << "    generic-tensors -dump-tokens <input_file>" << std::endl;
  std::cout << "    generic-tensors -dump-parsed <input_file>" << std::endl;
  std::cout << "    generic-tensors -dump-mlir [-clean-up] <input_file>" << std::endl;
  std::cout << "    generic-tensors -help" << std::endl;
}

int main( int argc, char **argv )
{
  if ( argc < 2 || argc > 4 ) {
    printUsage();
    return 1;
  }
  
  bool dumpTokens = false;
  bool dumpParsed = false;
  bool dumpMlir = false;
  bool cleanUp = false;
  
  if ( argc == 2 ) {
    std::string argument( argv[ 1 ] );
    if ( argument == "-help" ) {
      printUsage();
      return 0;
    } else {
      printUsage();
      return 1;
    }
  }
  
  std::string fileName;
  
  if ( argc == 3 ) {
    std::string option( argv[ 1 ] );
    if ( option == "-dump-tokens" ) {
      dumpTokens = true;
    } else if ( option == "-dump-parsed" ) {
      dumpParsed = true;
    } else if ( option == "-dump-mlir" ) {
      dumpMlir = true;
    } else {
      printUsage();
      return 1;
    }
    fileName = std::string( argv[ 2 ] );
  } else {
    std::string option1( argv[ 1 ] );
    std::string option2( argv[ 2 ] );
    if ( option1 != "-dump-mlir" || option2 != "-clean-up" ) {
      printUsage();
      return 1;
    }
    dumpMlir = true;
    cleanUp = true;
    fileName = std::string( argv[ 3 ] );
  }
  
  std::shared_ptr<gen_ten_compiler::TokenList> tokenList = gen_ten_compiler::Lexer::scanInputFile( fileName );
  if ( tokenList == nullptr ) {
    gen_ten::Logger::error() << "Could not open '" << fileName << "'" << std::endl;
    return 1;
  }
  
  if ( dumpTokens ) {
    for ( auto token : tokenList->tokens ) {
      std::cout << token << std::endl;
    }
    return 0;
  }
  
  gen_ten_compiler::Parser parser( *tokenList );
  std::shared_ptr<gen_ten_compiler::ProgramAST> programAST = parser.parseProgram();
  if ( programAST == nullptr ) {
    gen_ten::Logger::error() << "Could not parse '" << fileName << "'" << std::endl;
    return 1;
  }
  
  auto programIR = gen_ten_compiler::genTenGen( programAST );
  if ( programIR == nullptr ) {
    gen_ten::Logger::error() << "Could not generate GEN_TEN for '" << fileName << "'" << std::endl;
    return 1;
  }
  
  if ( dumpParsed ) {
    bool includeData = true;
    gen_ten::GenTenPrinter genTenPrinter( std::cout, includeData );
    genTenPrinter.printGenTen( programIR );
    return 0;
  }
  
  mlir::MLIRContext context;
  
  context.getOrLoadDialect<mlir::gen_ten::GenTenDialect>();
  
  mlir::OwningOpRef<mlir::ModuleOp> module = gen_ten_compiler::mlirGen( context, programIR );
  
  if ( cleanUp ) {
    mlir::PassManager pm( &context );
    applyPassManagerCLOptions( pm );
    mlir::OpPassManager &optPM = pm.nest<mlir::gen_ten::FunctionOp>();
    optPM.addPass( mlir::gen_ten::createGenTenCleanUpPass() );
    if ( mlir::failed( pm.run( *module ) ) ) {
      gen_ten::Logger::error() << "Could not clean up '" << fileName << "'" << std::endl;
      return 1;
    }
  }
  
  if ( dumpMlir ) {
    module->dump();
  }
  
  return gen_ten::Logger::hasErrors();
}
