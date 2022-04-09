//===-- GenTenPrinter.h - Printer for GEN_TEN ----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_PRINTER_H
#define MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_PRINTER_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace gen_ten_compiler {
struct NodeIR;
struct ProgramIR;
struct FunctionIR;
struct TensorIR;
struct SubtensorIR;
struct ParameterIR;
struct ResultIR;
struct VariableIR;
struct TensorValueIR;
struct TensorIntValueIR;
struct TensorFloatValueIR;
struct TensorStringValueIR;
struct AttributePairIR;
struct AttributeValueIR;
struct AttributeInt64ValueIR;
struct AttributeInt64ArrayValueIR;
struct AttributeFloatValueIR;
struct AttributeFloatArrayValueIR;
struct AttributeStringValueIR;
struct AttributeStringArrayValueIR;
struct AttributeVariableValueIR;
struct AttributeVariableArrayValueIR;
struct AttributeFunctionValueIR;
struct AttributeFunctionArrayValueIR;
struct OperationIR;
} // namspace gen_ten_compiler

namespace gen_ten {

class GenTenPrinter {
  std::ostream &os;
  bool shouldPrintData;
  
public:
  GenTenPrinter( std::ostream &os, bool shouldPrintData = false );
  
  std::string getTensorName( std::shared_ptr<gen_ten_compiler::NodeIR> node );
  
  void printGenTen( std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValueIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::ParameterIR> parameterIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::ResultIR> resultIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::VariableIR> variableIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::OperationIR> operationIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePairIR, std::string indent = "" );
  void printGenTen( std::vector<std::shared_ptr<gen_ten_compiler::AttributePairIR>> attributePairIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::ProgramIR> programIR, std::string indent = "" );
  void printGenTen( std::shared_ptr<gen_ten_compiler::NodeIR> nodeIR, std::string indent = "" );
};

} // namespace gen_ten

#endif // MLIR_DIALECT_GEN_TEN_IR_GEN_TEN_PRINTER_H
