//===- GenTenPrinter.cpp - Printer for GEN_TEN --------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Dialect/GenTen/IR/GenTenPrinter.h"

#include "Dialect/GenTen/IR/GenTenCompilerIR.hpp"

#include <iomanip>
#include <iostream>

namespace gen_ten {

GenTenPrinter::GenTenPrinter( std::ostream &os, bool shouldPrintData )
: os( os ), shouldPrintData( shouldPrintData )
{ }

std::string GenTenPrinter::getTensorName( std::shared_ptr<gen_ten_compiler::NodeIR> node ) {
  if ( node != nullptr ) {
    switch ( node->kind ) {
      case gen_ten_compiler::IR_parameter: {
        std::shared_ptr<gen_ten_compiler::ParameterIR> parameterIR = std::static_pointer_cast<gen_ten_compiler::ParameterIR>( node );
        return parameterIR->parameterName;
      }
        break;
      case gen_ten_compiler::IR_result: {
        std::shared_ptr<gen_ten_compiler::ResultIR> resultIR = std::static_pointer_cast<gen_ten_compiler::ResultIR>( node );
        return resultIR->resultName;
      }
        break;
      case gen_ten_compiler::IR_variable: {
        std::shared_ptr<gen_ten_compiler::VariableIR> variableIR = std::static_pointer_cast<gen_ten_compiler::VariableIR>( node );
        return variableIR->variableName;
      }
        break;
      default:
        break;
    }
  }
  return "";
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValueIR, std::string indent )
{
  if ( tensorValueIR != nullptr ) {
    switch ( tensorValueIR->childKind ) {
      case gen_ten_compiler::IR_tensorIntValue:
        if ( tensorValueIR->getTensorIntValueIR()->intValue.size() > 0 ) {
          os << " = [";
          
          if ( shouldPrintData ) {
            os << std::endl;
            auto tensorIntValueIR = tensorValueIR->getTensorIntValueIR();
            for ( uint64_t index = 0; index < tensorIntValueIR->intValue.size(); ++index ) {
              if ( index == 0 ) {
                os << indent << "    ";
              } else if ( index % 10 == 0 ) {
                os << "," << std::endl << indent << "    ";
              } else {
                os << ", ";
              }
              os << tensorIntValueIR->intValue[ index ];
              if ( index == tensorIntValueIR->intValue.size() - 1 ) {
                os << std::endl;
              }
            }
          }
          
          os << indent << "  ]";
        }
        break;
      case gen_ten_compiler::IR_tensorFloatValue:
        if ( tensorValueIR->getTensorFloatValueIR()->floatValue.size() > 0 ) {
          os << " = [";
          
          if ( shouldPrintData ) {
            os << std::endl;
            auto tensorFloatValueIR = tensorValueIR->getTensorFloatValueIR();
            for ( uint64_t index = 0; index < tensorFloatValueIR->floatValue.size(); ++index ) {
              if ( index == 0 ) {
                os << indent << "    ";
              } else if ( index % 10 == 0 ) {
                os << "," << std::endl << indent << "    ";
              } else {
                os << ", ";
              }
              os << std::setprecision( 10 );
              os << tensorFloatValueIR->floatValue[ index ];
              if ( index == tensorFloatValueIR->floatValue.size() - 1 ) {
                os << std::endl;
              }
            }
          }
          
          os << indent << "  ]";
        }
        break;
      case gen_ten_compiler::IR_tensorStringValue:
        if ( tensorValueIR->getTensorStringValueIR()->stringValue.size() > 0 ) {
          os << " = [";
          
          if ( shouldPrintData ) {
            os << std::endl;
            auto tensorStringValueIR = tensorValueIR->getTensorStringValueIR();
            for ( uint64_t index = 0; index < tensorStringValueIR->stringValue.size(); ++index ) {
              if ( index == 0 ) {
                os << indent << "    ";
              } else if ( index % 10 == 0 ) {
                os << "," << std::endl << indent << "    ";
              } else {
                os << ", ";
              }
              os << tensorStringValueIR->stringValue[ index ]; // TODO - Need to escape.
              if ( index == tensorStringValueIR->stringValue.size() - 1 ) {
                os << std::endl;
              }
            }
          }
          
          os << indent << "  ]";
        }
        break;
      default:
        break;
    }
  }
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR, std::string indent )
{
  os << gen_ten_compiler::getDataTypeName( tensorIR->dataType ) << "[";
  
  for ( uint64_t index = 0; index < tensorIR->dimension.size(); ++index ) {
    if ( index != 0 ) {
      os << "*";
    }
    os << tensorIR->dimension[ index ];
  }
  os << "]";
  
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::OperationIR> operationIR, std::string indent )
{
  os << indent;
  
  if ( operationIR->outputName.size() > 0 ) {
    os << "( ";
    for ( size_t i = 0; i < operationIR->outputName.size(); ++ i ) {
      if ( i != 0 ) {
        os << ", ";
      }
      os << operationIR->outputName[ i ];
    }
    os << " ) = ";
  }
  
  os << operationIR->operationName << "(";

  if ( operationIR->inputName.size() > 0 ) {
    os << " ";
    for ( size_t i = 0; i < operationIR->inputName.size(); ++ i ) {
      if ( i != 0 ) {
        os << ", ";
      }
      os << operationIR->inputName[ i ];
    }
    os << " ";
  }
  
  os << ")";
  
  printGenTen( operationIR->attributePair, indent );
  
  os << std::endl;
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePairIR, std::string indent )
{
  if ( attributePairIR != nullptr ) {
    os << "\"" << attributePairIR->attributeKey << "\" : ";
    
    switch ( attributePairIR->attributeValue->childKind ) {
      case gen_ten_compiler::IR_attributeInt64Value:
        os << attributePairIR->attributeValue->getAttributeInt64ValueIR()->int64Value;
        break;
      case gen_ten_compiler::IR_attributeInt64ArrayValue: {
        auto values = attributePairIR->attributeValue->getAttributeInt64ArrayValueIR()->int64Value;
        if ( values.size() > 0 ) {
          os << "[ ";
          uint64_t elementIndex = 0;
          while ( elementIndex < values.size() ) {
            if ( elementIndex != 0 ) {
              os << ", ";
            }
            os << values[ elementIndex ];
            elementIndex += 1;
          }
          os << " ]";
        }
      }
        break;
      case gen_ten_compiler::IR_attributeFloatValue:
        os << attributePairIR->attributeValue->getAttributeFloatValueIR()->floatValue;
        break;
      case gen_ten_compiler::IR_attributeFloatArrayValue: {
        auto values = attributePairIR->attributeValue->getAttributeFloatArrayValueIR()->floatValue;
        if ( values.size() > 0 ) {
          os << "[ ";
          uint64_t elementIndex = 0;
          while ( elementIndex < values.size() ) {
            if ( elementIndex != 0 ) {
              os << ", ";
            }
            os << values[ elementIndex ];
            elementIndex += 1;
          }
          os << " ]";
        }
      }
        break;
      case gen_ten_compiler::IR_attributeStringValue:
        os << "\"" << attributePairIR->attributeValue->getAttributeStringValueIR()->stringValue << "\"";
        break;
      case gen_ten_compiler::IR_attributeStringArrayValue: {
        auto values = attributePairIR->attributeValue->getAttributeStringArrayValueIR()->stringValue;
        if ( values.size() > 0 ) {
          os << "[ ";
          uint64_t elementIndex = 0;
          while ( elementIndex < values.size() ) {
            if ( elementIndex != 0 ) {
              os << ", ";
            }
            os << "\"" << values[ elementIndex ] << "\"";
            elementIndex += 1;
          }
          os << " ]";
        }
      }
        break;
      case gen_ten_compiler::IR_attributeVariableValue:
        os << attributePairIR->attributeValue->getAttributeVariableValueIR()->variableValue;
        break;
      case gen_ten_compiler::IR_attributeVariableArrayValue: {
        auto values = attributePairIR->attributeValue->getAttributeVariableArrayValueIR()->variableValue;
        if ( values.size() > 0 ) {
          os << "[ ";
          uint64_t elementIndex = 0;
          while ( elementIndex < values.size() ) {
            if ( elementIndex != 0 ) {
              os << ", ";
            }
            os << values[ elementIndex ];
            elementIndex += 1;
          }
          os << " ]";
        }
      }
        break;
      case gen_ten_compiler::IR_attributeFunctionValue:
        os << attributePairIR->attributeValue->getAttributeFunctionValueIR()->functionValue;
        break;
      case gen_ten_compiler::IR_attributeFunctionArrayValue: {
        auto values = attributePairIR->attributeValue->getAttributeFunctionArrayValueIR()->functionValue;
        if ( values.size() > 0 ) {
          os << "[ ";
          uint64_t elementIndex = 0;
          while ( elementIndex < values.size() ) {
            if ( elementIndex != 0 ) {
              os << ", ";
            }
            os << values[ elementIndex ];
            elementIndex += 1;
          }
          os << " ]";
        }
      }
        break;
      default:
        break;
    }
  }
}

void GenTenPrinter::printGenTen( std::vector<std::shared_ptr<gen_ten_compiler::AttributePairIR>> attributePairIR, std::string indent )
{
  if ( attributePairIR.size() > 0 ) {
    os << " [ ";
    
    uint64_t attributeIndex = 0;
    while ( attributeIndex < attributePairIR.size() ) {
      std::shared_ptr<gen_ten_compiler::AttributePairIR> pair = attributePairIR[ attributeIndex ];
      
      if ( pair != nullptr ) {
        if ( attributeIndex != 0 ) {
          os << ", ";
        }
        
        printGenTen( pair, indent );
      }
      
      attributeIndex += 1;
    }
    
    os << " ]";
  }
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::ProgramIR> programIR, std::string indent )
{
  std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR = programIR->function;
  
  printGenTen( functionIR, indent );
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::NodeIR> nodeIR, std::string indent )
{
  switch ( nodeIR->kind ) {
    case gen_ten_compiler::IR_tensorValue: {
      std::shared_ptr<gen_ten_compiler::TensorValueIR> tensorValueIR = std::static_pointer_cast<gen_ten_compiler::TensorValueIR>( nodeIR );
      printGenTen( tensorValueIR );
    }
      break;
    case gen_ten_compiler::IR_tensor: {
      std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR = std::static_pointer_cast<gen_ten_compiler::TensorIR>( nodeIR );
      printGenTen( tensorIR );
    }
      break;
    case gen_ten_compiler::IR_parameter: {
      std::shared_ptr<gen_ten_compiler::ParameterIR> parameterIR = std::static_pointer_cast<gen_ten_compiler::ParameterIR>( nodeIR );
      printGenTen( parameterIR );
    }
      break;
    case gen_ten_compiler::IR_result: {
      std::shared_ptr<gen_ten_compiler::ResultIR> resultIR = std::static_pointer_cast<gen_ten_compiler::ResultIR>( nodeIR );
      printGenTen( resultIR );
    }
      break;
    case gen_ten_compiler::IR_variable: {
      std::shared_ptr<gen_ten_compiler::VariableIR> variableIR = std::static_pointer_cast<gen_ten_compiler::VariableIR>( nodeIR );
      printGenTen( variableIR );
    }
      break;
    case gen_ten_compiler::IR_operation: {
      std::shared_ptr<gen_ten_compiler::OperationIR> operationIR = std::static_pointer_cast<gen_ten_compiler::OperationIR>( nodeIR );
      printGenTen( operationIR );
    }
      break;
    case gen_ten_compiler::IR_function: {
      std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR = std::static_pointer_cast<gen_ten_compiler::FunctionIR>( nodeIR );
      printGenTen( functionIR );
    }
      break;
    case gen_ten_compiler::IR_attributePair: {
      std::shared_ptr<gen_ten_compiler::AttributePairIR> attributePairIR = std::static_pointer_cast<gen_ten_compiler::AttributePairIR>( nodeIR );
      printGenTen( attributePairIR );
    }
      break;
    case gen_ten_compiler::IR_program: {
      std::shared_ptr<gen_ten_compiler::ProgramIR> programIR = std::static_pointer_cast<gen_ten_compiler::ProgramIR>( nodeIR );
      printGenTen( programIR );
    }
      break;
    default:
      break;
  }
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::ParameterIR> parameterIR, std::string indent )
{
  os << parameterIR->parameterName << ": ";
  printGenTen( parameterIR->tensor, indent );
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::ResultIR> resultIR, std::string indent )
{
  os << resultIR->resultName << ": ";
  printGenTen( resultIR->tensor, indent );
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::VariableIR> variableIR, std::string indent )
{
  os << indent << "  %var " << variableIR->variableName << ": ";
  if ( variableIR->tensor != nullptr ) {
    printGenTen( variableIR->tensor, indent );
  } else {
    os << "?[]";
  }
  printGenTen( variableIR->tensorValue, indent );
  os << std::endl;
}

void GenTenPrinter::printGenTen( std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR, std::string indent )
{
  os << indent << "%func " << functionIR->functionName << "(";
  
  for ( uint64_t index = 0; index < functionIR->parameter.size(); ++index ) {
    if ( index == 0 ) {
      os << " ";
    } else {
      os << ", ";
    }
    auto parameter = functionIR->parameter[ index ];
    printGenTen( parameter, indent );
    
    if ( index == functionIR->parameter.size() - 1 ) {
      os << " )" << std::endl;
    }
  }
  os << indent << "      -> (";
  
  for ( uint64_t index = 0; index < functionIR->result.size(); ++index ) {
    if ( index == 0 ) {
      os << " ";
    } else {
      os << ", ";
    }
    auto result = functionIR->result[ index ];
    printGenTen( result, indent );
    
    if ( index == functionIR->result.size() - 1 ) {
      os << " )" << std::endl;
    }
  }
  os << indent << "{" << std::endl;
  
  if ( functionIR->operation.size() > 0 ) {
    os << indent << "  " << std::endl;
    for ( auto operation : functionIR->operation ) {
      printGenTen( operation, indent + "  " );
    }
  }
  
  for ( auto function : functionIR->function ) {
    os << indent << "  " << std::endl;
    printGenTen( function, indent + "  " );
  }
  
  if ( functionIR->variable.size() > 0 ) {
    os << indent << "  " << std::endl;
    for ( auto variable : functionIR->variable ) {
      printGenTen( variable, indent );
    }
  }
  
  os << indent << "}" << std::endl;
}

} // namespace gen_ten
