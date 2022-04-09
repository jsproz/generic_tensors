//===- GenTenCompilerGenTenGen.cpp - MLIR Generation from a GenTen Binary AST -===//
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation from a Module AST
// for the GenTen Text language.
//
//===----------------------------------------------------------------------===//

#include "Frontend/GenTenCompiler/GenTenCompilerGenTenGen.h"

#include <iostream>
#include <vector>

#include "Frontend/GenTenCompiler/GenTenCompilerAST.hpp"
#include "Dialect/GenTen/IR/GenTenCompilerIR.hpp"

namespace {

struct GenTenCompilerGenTenGenImpl {
  
  GenTenCompilerGenTenGenImpl()
  { }
  
  std::shared_ptr<gen_ten_compiler::ProgramIR> genTenGen( std::shared_ptr<gen_ten_compiler::ProgramAST> program ) {
    if ( program->function == nullptr ) {
      return nullptr;
    }
    
    std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR = getFunction( program->function );
    if ( functionIR == nullptr )
      return nullptr;
    
    std::shared_ptr<gen_ten_compiler::ProgramIR> programIR( new gen_ten_compiler::ProgramIR( functionIR ) );
    
    return programIR;
  }
  
  std::shared_ptr<gen_ten_compiler::AttributePairIR> getAttributePair( std::shared_ptr<gen_ten_compiler::AttributePairAST> &attributePairAST ) {
    std::string key = attributePairAST->attributeKey->string->text;
    key = key.substr( 1, key.size() - 2 );
    
    auto attributeValueAST = attributePairAST->attributeValue;
    switch ( attributeValueAST->childKind ) {
      case gen_ten_compiler::ast_arrayValue: {
        auto arrayValueAST = attributeValueAST->getArrayValueAST();
        std::vector<int64_t> integerValues;
        std::vector<float> floatValues;
        std::vector<std::string> stringValues;
        for ( auto numberStringValueAST : arrayValueAST->numberStringValue ) {
          switch ( numberStringValueAST->childKind ) {
            case gen_ten_compiler::ast_numberValue: {
              auto numberValueAST = numberStringValueAST->getNumberValueAST();
              switch ( numberValueAST->childKind ) {
                case gen_ten_compiler::ast_integer: {
                  auto integerAST = numberValueAST->getIntegerAST();
                  integerValues.push_back( ( std::atoll( integerAST->text.c_str() ) ) );
                }
                  break;
                case gen_ten_compiler::ast_floatingPoint: {
                  auto floatingPointAST = numberValueAST->getFloatingPointAST();
                  floatValues.push_back( std::atof( floatingPointAST->text.c_str() ) );
                }
                  break;
                default:
                  break;
              }
            }
              break;
            case gen_ten_compiler::ast_stringValue: {
              auto stringValueAST = numberStringValueAST->getStringValueAST();
              std::string text = stringValueAST->string->text;
              text = text.substr( 1, text.size() - 2 );
              stringValues.push_back( text );
            }
              break;
            default:
              break;
          }
        }
        
        if ( stringValues.size() > 0 ) {
          std::shared_ptr<gen_ten_compiler::AttributeStringArrayValueIR> attributeStringArrayValue( new gen_ten_compiler::AttributeStringArrayValueIR( stringValues ) );
          std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeStringArrayValue ) );
          std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
          return attributePair;
        } else if ( floatValues.size() > 0 ) {
          std::shared_ptr<gen_ten_compiler::AttributeFloatArrayValueIR> attributeFloatArrayValue( new gen_ten_compiler::AttributeFloatArrayValueIR( floatValues ) );
          std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeFloatArrayValue ) );
          std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
          return attributePair;
        } else {
          std::shared_ptr<gen_ten_compiler::AttributeInt64ArrayValueIR> attributeInt64ArrayValue( new gen_ten_compiler::AttributeInt64ArrayValueIR( integerValues ) );
          std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeInt64ArrayValue ) );
          std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
          return attributePair;
        }
      }
        break;
      case gen_ten_compiler::ast_stringValue: {
        auto stringValueAST = attributeValueAST->getStringValueAST();
        std::string text = stringValueAST->string->text;
        text = text.substr( 1, text.size() - 2 );
        std::shared_ptr<gen_ten_compiler::AttributeStringValueIR> attributeStringValue( new gen_ten_compiler::AttributeStringValueIR( text ) );
        std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeStringValue ) );
        std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
        return attributePair;
      }
        break;
      case gen_ten_compiler::ast_numberValue: {
        auto numberValueAST = attributeValueAST->getNumberValueAST();
        switch ( numberValueAST->childKind ) {
          case gen_ten_compiler::ast_integer: {
            auto integerAST = numberValueAST->getIntegerAST();
            std::shared_ptr<gen_ten_compiler::AttributeInt64ValueIR> attributeInt64Value( new gen_ten_compiler::AttributeInt64ValueIR( std::atoll( integerAST->text.c_str() ) ) );
            std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeInt64Value ) );
            std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
            return attributePair;
          }
            break;
          case gen_ten_compiler::ast_floatingPoint: {
            auto floatingPointAST = numberValueAST->getFloatingPointAST();
            std::shared_ptr<gen_ten_compiler::AttributeFloatValueIR> attributeFloatValue( new gen_ten_compiler::AttributeFloatValueIR( std::atof( floatingPointAST->text.c_str() ) ) );
            std::shared_ptr<gen_ten_compiler::AttributeValueIR> attributeValue( new gen_ten_compiler::AttributeValueIR( attributeFloatValue ) );
            std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePair( new gen_ten_compiler::AttributePairIR( key, attributeValue ) );
            return attributePair;
          }
            break;
          default:
            break;
        }
      }
        break;
      default:
        break;
    }
    
    return nullptr;
  }
  
  std::shared_ptr<gen_ten_compiler::OperationIR> getOperation( std::shared_ptr<gen_ten_compiler::OperationAST> operation ) {
    std::string operationName = operation->operationName->identifier->text;
    
    std::vector<std::string> resultName;
    auto resultPhrase = operation->resultPhrase;
    if ( resultPhrase != nullptr ) {
      for ( auto output : resultPhrase->output ) {
        resultName.push_back( output->identifier->text );
      }
    }
    
    std::vector<std::string> argumentName;
    auto argumentPhrase = operation->argumentPhrase;
    for ( auto input : argumentPhrase->input ) {
      argumentName.push_back( input->identifier->text );
    }
    
    std::vector<std::shared_ptr<gen_ten_compiler::AttributePairIR>> attributePairs;
    if ( operation->attributes != nullptr ) {
      auto attributesAST = operation->attributes;
      for ( auto attributePair : attributesAST->attributePair ) {
        std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePairIR = getAttributePair( attributePair );
        if ( attributePairIR != nullptr ) {
          attributePairs.push_back( attributePairIR );
        }
      }
    }
    
    std::shared_ptr<gen_ten_compiler::OperationIR> operationIR( new gen_ten_compiler::OperationIR( resultName,
                                                                                                  operationName,
                                                                                                  argumentName,
                                                                                                  attributePairs ) );
    
    return operationIR;
  }
  
  void getTensorFloatValues( std::shared_ptr<gen_ten_compiler::FloatingPointAST> floatingPoint, std::vector<float> &floatData ) {
    float floatValue = std::stod( floatingPoint->text );
    floatData.push_back( floatValue );
  }
  
  void getTensorFloatValues( std::shared_ptr<gen_ten_compiler::IntegerAST> integer, std::vector<float> &floatData ) {
    int64_t intValue = std::stol( integer->text );
    floatData.push_back( intValue );
  }
  
  void getTensorFloatValues( std::shared_ptr<gen_ten_compiler::NumberValueAST> numberValue, std::vector<float> &floatData ) {
    switch ( numberValue->childKind ) {
      case gen_ten_compiler::ast_integer: {
        std::shared_ptr<gen_ten_compiler::IntegerAST> integer = numberValue->getIntegerAST();
        getTensorFloatValues( integer, floatData );
      }
        break;
      case gen_ten_compiler::ast_floatingPoint: {
        std::shared_ptr<gen_ten_compiler::FloatingPointAST> floatingPoint = numberValue->getFloatingPointAST();
        getTensorFloatValues( floatingPoint, floatData );
      }
        break;
      default:
        break;
    }
  }
  
  void getTensorFloatValues( std::shared_ptr<gen_ten_compiler::TensorValueAST> tensorValue, std::vector<float> &floatData ) {
    for ( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue : tensorValue->constantValue ) {
      getTensorFloatValues( constantValue, floatData );
    }
  }
  
  void getTensorFloatValues( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue, std::vector<float> &floatData ) {
    switch ( constantValue->childKind ) {
      case gen_ten_compiler::ast_numberValue: {
        std::shared_ptr<gen_ten_compiler::NumberValueAST> numberValue = constantValue->getNumberValueAST();
        getTensorFloatValues( numberValue, floatData );
      }
        break;
      case gen_ten_compiler::ast_tensorValue: {
        std::shared_ptr<gen_ten_compiler::TensorValueAST> tensorValue = constantValue->getTensorValueAST();
        getTensorFloatValues( tensorValue, floatData );
      }
        break;
      default:
        break;
    }
  }
  
  void getTensorIntValues( std::shared_ptr<gen_ten_compiler::IntegerAST> integer, std::vector<int64_t> &intData ) {
    int64_t intValue = std::stol( integer->text );
    intData.push_back( intValue );
  }
  
  void getTensorIntValues( std::shared_ptr<gen_ten_compiler::NumberValueAST> numberValue, std::vector<int64_t> &intData ) {
    switch ( numberValue->childKind ) {
      case gen_ten_compiler::ast_integer: {
        std::shared_ptr<gen_ten_compiler::IntegerAST> integer = numberValue->getIntegerAST();
        getTensorIntValues( integer, intData );
      }
        break;
      default:
        break;
    }
  }
  
  void getTensorIntValues( std::shared_ptr<gen_ten_compiler::TensorValueAST> tensorValue, std::vector<int64_t> &intData ) {
    for ( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue : tensorValue->constantValue ) {
      getTensorIntValues( constantValue, intData );
    }
  }
  
  void getTensorIntValues( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue, std::vector<int64_t> &floatData ) {
    switch ( constantValue->childKind ) {
      case gen_ten_compiler::ast_numberValue: {
        std::shared_ptr<gen_ten_compiler::NumberValueAST> numberValue = constantValue->getNumberValueAST();
        getTensorIntValues( numberValue, floatData );
      }
        break;
      case gen_ten_compiler::ast_tensorValue: {
        std::shared_ptr<gen_ten_compiler::TensorValueAST> tensorValue = constantValue->getTensorValueAST();
        getTensorIntValues( tensorValue, floatData );
      }
        break;
      default:
        break;
    }
  }
  
  std::shared_ptr<gen_ten_compiler::TensorValueIR> getTensorValue( std::shared_ptr<gen_ten_compiler::TensorValueAST> tensorValue, bool isFloatingPoint ) {
    if ( isFloatingPoint ) {
      std::vector<float> floatData;
      for ( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue : tensorValue->constantValue ) {
        getTensorFloatValues( constantValue, floatData );
      }
      std::shared_ptr<gen_ten_compiler::TensorFloatValueIR> tensorFloatValue( new gen_ten_compiler::TensorFloatValueIR( floatData ) );
      std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValueIR( new gen_ten_compiler::TensorValueIR( tensorFloatValue ) );
      return tensorValueIR;
    } else {
      std::vector<int64_t> intData;
      for ( std::shared_ptr<gen_ten_compiler::ConstantValueAST> constantValue : tensorValue->constantValue ) {
        getTensorIntValues( constantValue, intData );
      }
      std::shared_ptr<gen_ten_compiler::TensorIntValueIR> tensorIntValue( new gen_ten_compiler::TensorIntValueIR( intData ) );
      std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValueIR( new gen_ten_compiler::TensorValueIR( tensorIntValue ) );
      return tensorValueIR;
    }
    return nullptr;
  }
  
  std::shared_ptr<gen_ten_compiler::TensorIR> getTensor( std::shared_ptr<gen_ten_compiler::TensorAST> tensor ) {
    std::vector<int64_t> dimensions;
    
    for ( std::shared_ptr<gen_ten_compiler::DimensionAST> dimension : tensor->dimension ) {
      switch ( dimension->childKind ) {
        case gen_ten_compiler::ast_integerDimension: {
          std::shared_ptr<gen_ten_compiler::IntegerDimensionAST> integerDimension = dimension->getIntegerDimensionAST();
          dimensions.push_back( std::stoi( integerDimension->integer->text ) );
        }
          break;
        default:
          break;
      }
    }
    
    gen_ten_compiler::DataTypeIR dataType = gen_ten_compiler::DT_unknown;
    
    switch( tensor->dataType->childKind ) {
      case gen_ten_compiler::ast_dataTypeFloat32:
        dataType = gen_ten_compiler::DT_float32;
        break;
      case gen_ten_compiler::ast_dataTypeInt64:
        dataType = gen_ten_compiler::DT_int64;
        break;
      default:
        break;
    }
    
    std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR( new gen_ten_compiler::TensorIR( dimensions, dataType ) );
    return tensorIR;
  }
  
  std::shared_ptr<gen_ten_compiler::FunctionIR> getFunction( std::shared_ptr<gen_ten_compiler::FunctionAST> function ) {
    std::string functionName = function->functionName->identifier->text;
    
    std::vector<std::shared_ptr<gen_ten_compiler::ParameterIR>> parameters;
    auto inputPhrase = function->inputPhrase;
    if ( inputPhrase != nullptr ) {
      for ( auto parameter : inputPhrase->parameter ) {
        std::string parameterName = parameter->parameterName->identifier->text;
        std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR = getTensor( parameter->tensor );
        std::shared_ptr<gen_ten_compiler::ParameterIR> parameterIR( new gen_ten_compiler::ParameterIR( parameterName, tensorIR ) );
        parameters.push_back( parameterIR );
      }
    }
    
    std::vector<std::shared_ptr<gen_ten_compiler::ResultIR>> results;
    auto outputPhrase = function->outputPhrase;
    if ( outputPhrase != nullptr ) {
      for ( auto result : outputPhrase->result ) {
        std::string resultName = result->resultName->identifier->text;
        std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR = getTensor( result->tensor );
        std::shared_ptr<gen_ten_compiler::ResultIR> resultIR( new gen_ten_compiler::ResultIR( resultName, tensorIR ) );
        results.push_back( resultIR );
      }
    }
    
    std::vector<std::shared_ptr<gen_ten_compiler::VariableIR>> variables;
    for ( auto statement : function->statement ) {
      if ( statement->childKind == gen_ten_compiler::ast_variable ) {
        auto variable = statement->getVariableAST();
        std::string variableName = variable->variableName->identifier->text;
        std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR = getTensor( variable->tensor );
        
        std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValue;
        if ( variable->initializer != nullptr ) {
          tensorValue = getTensorValue( variable->initializer->tensorValue, tensorIR->dataType == gen_ten_compiler::DT_float32 );
        }
        
        std::shared_ptr<gen_ten_compiler::VariableIR> variableIR( new gen_ten_compiler::VariableIR( variableName, tensorIR, tensorValue ) );
        variables.push_back( variableIR );
      }
    }
    
    std::vector<std::shared_ptr<gen_ten_compiler::FunctionIR>> subfunctions;
    for ( auto statement : function->statement ) {
      if ( statement->childKind == gen_ten_compiler::ast_function ) {
        auto subfunction = statement->getFunctionAST();
        std::shared_ptr<gen_ten_compiler::FunctionIR> subfunctionIR = getFunction( subfunction );
        subfunctions.push_back( subfunctionIR );
      }
    }
    
    std::vector<std::shared_ptr<gen_ten_compiler::OperationIR>> dependencies;
    for ( auto statement : function->statement ) {
      if ( statement->childKind == gen_ten_compiler::ast_operation ) {
        auto operation = statement->getOperationAST();
        std::shared_ptr<gen_ten_compiler::OperationIR> operationIR = getOperation( operation );
        dependencies.push_back( operationIR );
      }
    }
    
    std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR( new gen_ten_compiler::FunctionIR( functionName,
                                                                                      parameters,
                                                                                      results,
                                                                                      variables,
                                                                                      subfunctions,
                                                                                      dependencies ) );
    
    return functionIR;
  }

};

} // namespace

namespace gen_ten_compiler {

std::shared_ptr<gen_ten_compiler::ProgramIR> genTenGen( std::shared_ptr<gen_ten_compiler::ProgramAST> program ) {
  GenTenCompilerGenTenGenImpl genTenCompilerGenTenGenImpl;
  std::shared_ptr<gen_ten_compiler::ProgramIR> programIR = genTenCompilerGenTenGenImpl.genTenGen( program );
  return programIR;
}

} // namespace gen_ten_compiler
