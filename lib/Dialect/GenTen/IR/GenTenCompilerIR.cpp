//===- GenTenCompilerIR.cpp - IR for the GenTen language -*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file implements an intermediate representation for the GenTen language.
//
// This file was generated.
//
//===----------------------------------------------------------------------===//

#include "Dialect/GenTen/IR/GenTenCompilerIR.hpp"

#include <algorithm>
#include <vector>

namespace gen_ten_compiler {

const char *getDataTypeName( DataTypeIR dataType ) {
  switch ( dataType ) {
    case DT_unknown:
      return "?";
    case DT_float32:
      return "Float32";
    case DT_int64:
      return "Int64";
    case DT_string:
      return "String";
  }
}

const char *getDataTypeEnumName( DataTypeIR dataType ) {
  switch ( dataType ) {
    case DT_unknown:
      return "DT_unknown";
    case DT_float32:
      return "DT_float32";
    case DT_int64:
      return "DT_int64";
    case DT_string:
      return "DT_string";
  }
}

struct DataTypeTableEntry {
  const char *literal;
  DataTypeIR enumerator;
};

static DataTypeTableEntry dataTypeTable[] = {
  { "?", DT_unknown },
  { "Float32", DT_float32 },
  { "Int64", DT_int64 },
  { "String", DT_string }
};

DataTypeIR getDataType( std::string literal )
{
  int tableSize = sizeof( dataTypeTable ) / sizeof( DataTypeTableEntry );
  int index = 0;
  while ( index < tableSize ) {
    if ( dataTypeTable[ index ].literal == literal ) {
      return dataTypeTable[ index ].enumerator;
    }
    index += 1;
  }
  
  return DT_unknown;
}

std::string getDataTypeLiteral( DataTypeIR enumerator )
{
  int tableSize = sizeof( dataTypeTable ) / sizeof( DataTypeTableEntry );
  int index = 0;
  while ( index < tableSize ) {
    if ( dataTypeTable[ index ].enumerator == enumerator ) {
      return dataTypeTable[ index ].literal;
    }
    index += 1;
  }
  
  return dataTypeTable[ 0 ].literal;
}

NodeIR::NodeIR( NodeIRKind kind )
: kind( kind )
{ }

ProgramIR::ProgramIR( std::shared_ptr<FunctionIR> function )
: NodeIR( IR_program ), function( function )
{ }

FunctionIR::FunctionIR( std::string functionName,
          std::vector<std::shared_ptr<ParameterIR>> parameter,
          std::vector<std::shared_ptr<ResultIR>> result,
          std::vector<std::shared_ptr<VariableIR>> variable,
          std::vector<std::shared_ptr<FunctionIR>> function,
          std::vector<std::shared_ptr<OperationIR>> operation )
: NodeIR( IR_function ), functionName( functionName ), parameter( parameter ), result( result ), variable( variable ), function( function ), operation( operation )
{ }

TensorIR::TensorIR( std::vector<int64_t> dimension,
          DataTypeIR dataType )
: NodeIR( IR_tensor ), dimension( dimension ), dataType( dataType )
{ }

SubtensorIR::SubtensorIR( std::vector<int64_t> dimension,
          std::vector<int64_t> coordinate,
          std::shared_ptr<TensorIR> tensor )
: NodeIR( IR_subtensor ), dimension( dimension ), coordinate( coordinate ), tensor( tensor )
{ }

ParameterIR::ParameterIR( std::string parameterName,
          std::shared_ptr<TensorIR> tensor )
: NodeIR( IR_parameter ), parameterName( parameterName ), tensor( tensor )
{ }

ResultIR::ResultIR( std::string resultName,
          std::shared_ptr<TensorIR> tensor )
: NodeIR( IR_result ), resultName( resultName ), tensor( tensor )
{ }

VariableIR::VariableIR( std::string variableName,
          std::shared_ptr<TensorIR> tensor,
          std::shared_ptr<TensorValueIR> tensorValue )
: NodeIR( IR_variable ), variableName( variableName ), tensor( tensor ), tensorValue( tensorValue )
{ }

TensorValueIR::TensorValueIR()
: NodeIR( IR_tensorValue ), childKind( IR_unknown ), child( nullptr )
{ }

TensorValueIR::TensorValueIR( std::shared_ptr<TensorIntValueIR> tensorIntValue )
: NodeIR( IR_tensorValue ), childKind( IR_tensorIntValue ), child( tensorIntValue )
{ }

TensorValueIR::TensorValueIR( std::shared_ptr<TensorFloatValueIR> tensorFloatValue )
: NodeIR( IR_tensorValue ), childKind( IR_tensorFloatValue ), child( tensorFloatValue )
{ }

TensorValueIR::TensorValueIR( std::shared_ptr<TensorStringValueIR> tensorStringValue )
: NodeIR( IR_tensorValue ), childKind( IR_tensorStringValue ), child( tensorStringValue )
{ }

std::shared_ptr<TensorIntValueIR> TensorValueIR::getTensorIntValueIR()
{
  return ( childKind == IR_tensorIntValue ) ? std::static_pointer_cast<TensorIntValueIR>(child) : nullptr;
}

std::shared_ptr<TensorFloatValueIR> TensorValueIR::getTensorFloatValueIR()
{
  return ( childKind == IR_tensorFloatValue ) ? std::static_pointer_cast<TensorFloatValueIR>(child) : nullptr;
}

std::shared_ptr<TensorStringValueIR> TensorValueIR::getTensorStringValueIR()
{
  return ( childKind == IR_tensorStringValue ) ? std::static_pointer_cast<TensorStringValueIR>(child) : nullptr;
}

TensorIntValueIR::TensorIntValueIR( std::vector<int64_t> intValue )
: NodeIR( IR_tensorIntValue ), intValue( intValue )
{ }

TensorFloatValueIR::TensorFloatValueIR( std::vector<float> floatValue )
: NodeIR( IR_tensorFloatValue ), floatValue( floatValue )
{ }

TensorStringValueIR::TensorStringValueIR( std::vector<std::string> stringValue )
: NodeIR( IR_tensorStringValue ), stringValue( stringValue )
{ }

AttributePairIR::AttributePairIR( std::string attributeKey,
          std::shared_ptr<AttributeValueIR> attributeValue )
: NodeIR( IR_attributePair ), attributeKey( attributeKey ), attributeValue( attributeValue )
{ }

AttributeValueIR::AttributeValueIR()
: NodeIR( IR_attributeValue ), childKind( IR_unknown ), child( nullptr )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeInt64ValueIR> attributeInt64Value )
: NodeIR( IR_attributeValue ), childKind( IR_attributeInt64Value ), child( attributeInt64Value )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeInt64ArrayValueIR> attributeInt64ArrayValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeInt64ArrayValue ), child( attributeInt64ArrayValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeFloatValueIR> attributeFloatValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeFloatValue ), child( attributeFloatValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeFloatArrayValueIR> attributeFloatArrayValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeFloatArrayValue ), child( attributeFloatArrayValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeStringValueIR> attributeStringValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeStringValue ), child( attributeStringValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeStringArrayValueIR> attributeStringArrayValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeStringArrayValue ), child( attributeStringArrayValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeVariableValueIR> attributeVariableValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeVariableValue ), child( attributeVariableValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeVariableArrayValueIR> attributeVariableArrayValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeVariableArrayValue ), child( attributeVariableArrayValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeFunctionValueIR> attributeFunctionValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeFunctionValue ), child( attributeFunctionValue )
{ }

AttributeValueIR::AttributeValueIR( std::shared_ptr<AttributeFunctionArrayValueIR> attributeFunctionArrayValue )
: NodeIR( IR_attributeValue ), childKind( IR_attributeFunctionArrayValue ), child( attributeFunctionArrayValue )
{ }

std::shared_ptr<AttributeInt64ValueIR> AttributeValueIR::getAttributeInt64ValueIR()
{
  return ( childKind == IR_attributeInt64Value ) ? std::static_pointer_cast<AttributeInt64ValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeInt64ArrayValueIR> AttributeValueIR::getAttributeInt64ArrayValueIR()
{
  return ( childKind == IR_attributeInt64ArrayValue ) ? std::static_pointer_cast<AttributeInt64ArrayValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeFloatValueIR> AttributeValueIR::getAttributeFloatValueIR()
{
  return ( childKind == IR_attributeFloatValue ) ? std::static_pointer_cast<AttributeFloatValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeFloatArrayValueIR> AttributeValueIR::getAttributeFloatArrayValueIR()
{
  return ( childKind == IR_attributeFloatArrayValue ) ? std::static_pointer_cast<AttributeFloatArrayValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeStringValueIR> AttributeValueIR::getAttributeStringValueIR()
{
  return ( childKind == IR_attributeStringValue ) ? std::static_pointer_cast<AttributeStringValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeStringArrayValueIR> AttributeValueIR::getAttributeStringArrayValueIR()
{
  return ( childKind == IR_attributeStringArrayValue ) ? std::static_pointer_cast<AttributeStringArrayValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeVariableValueIR> AttributeValueIR::getAttributeVariableValueIR()
{
  return ( childKind == IR_attributeVariableValue ) ? std::static_pointer_cast<AttributeVariableValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeVariableArrayValueIR> AttributeValueIR::getAttributeVariableArrayValueIR()
{
  return ( childKind == IR_attributeVariableArrayValue ) ? std::static_pointer_cast<AttributeVariableArrayValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeFunctionValueIR> AttributeValueIR::getAttributeFunctionValueIR()
{
  return ( childKind == IR_attributeFunctionValue ) ? std::static_pointer_cast<AttributeFunctionValueIR>(child) : nullptr;
}

std::shared_ptr<AttributeFunctionArrayValueIR> AttributeValueIR::getAttributeFunctionArrayValueIR()
{
  return ( childKind == IR_attributeFunctionArrayValue ) ? std::static_pointer_cast<AttributeFunctionArrayValueIR>(child) : nullptr;
}

AttributeInt64ValueIR::AttributeInt64ValueIR( int64_t int64Value )
: NodeIR( IR_attributeInt64Value ), int64Value( int64Value )
{ }

AttributeInt64ArrayValueIR::AttributeInt64ArrayValueIR( std::vector<int64_t> int64Value )
: NodeIR( IR_attributeInt64ArrayValue ), int64Value( int64Value )
{ }

AttributeFloatValueIR::AttributeFloatValueIR( float floatValue )
: NodeIR( IR_attributeFloatValue ), floatValue( floatValue )
{ }

AttributeFloatArrayValueIR::AttributeFloatArrayValueIR( std::vector<float> floatValue )
: NodeIR( IR_attributeFloatArrayValue ), floatValue( floatValue )
{ }

AttributeStringValueIR::AttributeStringValueIR( std::string stringValue )
: NodeIR( IR_attributeStringValue ), stringValue( stringValue )
{ }

AttributeStringArrayValueIR::AttributeStringArrayValueIR( std::vector<std::string> stringValue )
: NodeIR( IR_attributeStringArrayValue ), stringValue( stringValue )
{ }

AttributeVariableValueIR::AttributeVariableValueIR( std::string variableValue )
: NodeIR( IR_attributeVariableValue ), variableValue( variableValue )
{ }

AttributeVariableArrayValueIR::AttributeVariableArrayValueIR( std::vector<std::string> variableValue )
: NodeIR( IR_attributeVariableArrayValue ), variableValue( variableValue )
{ }

AttributeFunctionValueIR::AttributeFunctionValueIR( std::string functionValue )
: NodeIR( IR_attributeFunctionValue ), functionValue( functionValue )
{ }

AttributeFunctionArrayValueIR::AttributeFunctionArrayValueIR( std::vector<std::string> functionValue )
: NodeIR( IR_attributeFunctionArrayValue ), functionValue( functionValue )
{ }

OperationIR::OperationIR( std::vector<std::string> outputName,
          std::string operationName,
          std::vector<std::string> inputName,
          std::vector<std::shared_ptr<AttributePairIR>> attributePair )
: NodeIR( IR_operation ), outputName( outputName ), operationName( operationName ), inputName( inputName ), attributePair( attributePair )
{ }

std::shared_ptr<AttributePairIR> OperationIR::getAttributePair( std::string keyName )
{
  std::vector<std::shared_ptr<AttributePairIR>>::iterator it = std::find_if( attributePair.begin(), attributePair.end(),
      [&]( const std::shared_ptr<AttributePairIR> & pair ) -> bool { return pair->attributeKey == keyName; } );
  if ( it == attributePair.end() ) {
    return nullptr;
  } else {
    return *it;
  }
}

bool OperationIR::hasAttributePair( std::string keyName )
{
  std::shared_ptr<AttributePairIR> attributePair = getAttributePair( keyName );
  return attributePair != nullptr;
}

} // namespace gen_ten_compiler
