//===- GenTenCompilerIR.h - IR for the GenTen language -*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file defines an intermediate representation for the GenTen language.
//
// This file was generated.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_IR_HPP
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_IR_HPP

#include <memory>
#include <string>
#include <vector>

namespace gen_ten_compiler {

enum DataTypeIR {
  DT_unknown,
  DT_float32,
  DT_int64,
  DT_string
};

const char *getDataTypeName( DataTypeIR dataType );
const char *getDataTypeEnumName( DataTypeIR dataType );
DataTypeIR getDataType( std::string literal );
std::string getDataTypeLiteral( DataTypeIR enumerator );

enum NodeIRKind {
  IR_unknown,
  IR_program,
  IR_function,
  IR_tensor,
  IR_subtensor,
  IR_parameter,
  IR_result,
  IR_variable,
  IR_tensorValue,
  IR_tensorIntValue,
  IR_tensorFloatValue,
  IR_tensorStringValue,
  IR_attributePair,
  IR_attributeValue,
  IR_attributeInt64Value,
  IR_attributeInt64ArrayValue,
  IR_attributeFloatValue,
  IR_attributeFloatArrayValue,
  IR_attributeStringValue,
  IR_attributeStringArrayValue,
  IR_attributeVariableValue,
  IR_attributeVariableArrayValue,
  IR_attributeFunctionValue,
  IR_attributeFunctionArrayValue,
  IR_operation
};

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

struct NodeIR {
  NodeIRKind kind;
  NodeIR( NodeIRKind kind );
};

struct ProgramIR : public NodeIR {
  std::shared_ptr<FunctionIR> function;
  
  ProgramIR( std::shared_ptr<FunctionIR> function );
};

struct FunctionIR : public NodeIR {
  std::string functionName;
  std::vector<std::shared_ptr<ParameterIR>> parameter;
  std::vector<std::shared_ptr<ResultIR>> result;
  std::vector<std::shared_ptr<VariableIR>> variable;
  std::vector<std::shared_ptr<FunctionIR>> function;
  std::vector<std::shared_ptr<OperationIR>> operation;
  
  FunctionIR( std::string functionName,
          std::vector<std::shared_ptr<ParameterIR>> parameter,
          std::vector<std::shared_ptr<ResultIR>> result,
          std::vector<std::shared_ptr<VariableIR>> variable,
          std::vector<std::shared_ptr<FunctionIR>> function,
          std::vector<std::shared_ptr<OperationIR>> operation );
};

struct TensorIR : public NodeIR {
  std::vector<int64_t> dimension;
  DataTypeIR dataType;
  
  TensorIR( std::vector<int64_t> dimension,
          DataTypeIR dataType );
};

struct SubtensorIR : public NodeIR {
  std::vector<int64_t> dimension;
  std::vector<int64_t> coordinate;
  std::shared_ptr<TensorIR> tensor;
  
  SubtensorIR( std::vector<int64_t> dimension,
          std::vector<int64_t> coordinate,
          std::shared_ptr<TensorIR> tensor );
};

struct ParameterIR : public NodeIR {
  std::string parameterName;
  std::shared_ptr<TensorIR> tensor;
  
  ParameterIR( std::string parameterName,
          std::shared_ptr<TensorIR> tensor );
};

struct ResultIR : public NodeIR {
  std::string resultName;
  std::shared_ptr<TensorIR> tensor;
  
  ResultIR( std::string resultName,
          std::shared_ptr<TensorIR> tensor );
};

struct VariableIR : public NodeIR {
  std::string variableName;
  std::shared_ptr<TensorIR> tensor;
  std::shared_ptr<TensorValueIR> tensorValue;
  
  VariableIR( std::string variableName,
          std::shared_ptr<TensorIR> tensor,
          std::shared_ptr<TensorValueIR> tensorValue );
};

struct TensorValueIR : public NodeIR {
  NodeIRKind childKind;
  std::shared_ptr<NodeIR> child;
  
  TensorValueIR();
  TensorValueIR( std::shared_ptr<TensorIntValueIR> tensorIntValue );
  TensorValueIR( std::shared_ptr<TensorFloatValueIR> tensorFloatValue );
  TensorValueIR( std::shared_ptr<TensorStringValueIR> tensorStringValue );
  std::shared_ptr<TensorIntValueIR> getTensorIntValueIR();
  std::shared_ptr<TensorFloatValueIR> getTensorFloatValueIR();
  std::shared_ptr<TensorStringValueIR> getTensorStringValueIR();
};

struct TensorIntValueIR : public NodeIR {
  std::vector<int64_t> intValue;
  
  TensorIntValueIR( std::vector<int64_t> intValue );
};

struct TensorFloatValueIR : public NodeIR {
  std::vector<float> floatValue;
  
  TensorFloatValueIR( std::vector<float> floatValue );
};

struct TensorStringValueIR : public NodeIR {
  std::vector<std::string> stringValue;
  
  TensorStringValueIR( std::vector<std::string> stringValue );
};

struct AttributePairIR : public NodeIR {
  std::string attributeKey;
  std::shared_ptr<AttributeValueIR> attributeValue;
  
  AttributePairIR( std::string attributeKey,
          std::shared_ptr<AttributeValueIR> attributeValue );
};

struct AttributeValueIR : public NodeIR {
  NodeIRKind childKind;
  std::shared_ptr<NodeIR> child;
  
  AttributeValueIR();
  AttributeValueIR( std::shared_ptr<AttributeInt64ValueIR> attributeInt64Value );
  AttributeValueIR( std::shared_ptr<AttributeInt64ArrayValueIR> attributeInt64ArrayValue );
  AttributeValueIR( std::shared_ptr<AttributeFloatValueIR> attributeFloatValue );
  AttributeValueIR( std::shared_ptr<AttributeFloatArrayValueIR> attributeFloatArrayValue );
  AttributeValueIR( std::shared_ptr<AttributeStringValueIR> attributeStringValue );
  AttributeValueIR( std::shared_ptr<AttributeStringArrayValueIR> attributeStringArrayValue );
  AttributeValueIR( std::shared_ptr<AttributeVariableValueIR> attributeVariableValue );
  AttributeValueIR( std::shared_ptr<AttributeVariableArrayValueIR> attributeVariableArrayValue );
  AttributeValueIR( std::shared_ptr<AttributeFunctionValueIR> attributeFunctionValue );
  AttributeValueIR( std::shared_ptr<AttributeFunctionArrayValueIR> attributeFunctionArrayValue );
  std::shared_ptr<AttributeInt64ValueIR> getAttributeInt64ValueIR();
  std::shared_ptr<AttributeInt64ArrayValueIR> getAttributeInt64ArrayValueIR();
  std::shared_ptr<AttributeFloatValueIR> getAttributeFloatValueIR();
  std::shared_ptr<AttributeFloatArrayValueIR> getAttributeFloatArrayValueIR();
  std::shared_ptr<AttributeStringValueIR> getAttributeStringValueIR();
  std::shared_ptr<AttributeStringArrayValueIR> getAttributeStringArrayValueIR();
  std::shared_ptr<AttributeVariableValueIR> getAttributeVariableValueIR();
  std::shared_ptr<AttributeVariableArrayValueIR> getAttributeVariableArrayValueIR();
  std::shared_ptr<AttributeFunctionValueIR> getAttributeFunctionValueIR();
  std::shared_ptr<AttributeFunctionArrayValueIR> getAttributeFunctionArrayValueIR();
};

struct AttributeInt64ValueIR : public NodeIR {
  int64_t int64Value;
  
  AttributeInt64ValueIR( int64_t int64Value );
};

struct AttributeInt64ArrayValueIR : public NodeIR {
  std::vector<int64_t> int64Value;
  
  AttributeInt64ArrayValueIR( std::vector<int64_t> int64Value );
};

struct AttributeFloatValueIR : public NodeIR {
  float floatValue;
  
  AttributeFloatValueIR( float floatValue );
};

struct AttributeFloatArrayValueIR : public NodeIR {
  std::vector<float> floatValue;
  
  AttributeFloatArrayValueIR( std::vector<float> floatValue );
};

struct AttributeStringValueIR : public NodeIR {
  std::string stringValue;
  
  AttributeStringValueIR( std::string stringValue );
};

struct AttributeStringArrayValueIR : public NodeIR {
  std::vector<std::string> stringValue;
  
  AttributeStringArrayValueIR( std::vector<std::string> stringValue );
};

struct AttributeVariableValueIR : public NodeIR {
  std::string variableValue;
  
  AttributeVariableValueIR( std::string variableValue );
};

struct AttributeVariableArrayValueIR : public NodeIR {
  std::vector<std::string> variableValue;
  
  AttributeVariableArrayValueIR( std::vector<std::string> variableValue );
};

struct AttributeFunctionValueIR : public NodeIR {
  std::string functionValue;
  
  AttributeFunctionValueIR( std::string functionValue );
};

struct AttributeFunctionArrayValueIR : public NodeIR {
  std::vector<std::string> functionValue;
  
  AttributeFunctionArrayValueIR( std::vector<std::string> functionValue );
};

struct OperationIR : public NodeIR {
  std::vector<std::string> outputName;
  std::string operationName;
  std::vector<std::string> inputName;
  std::vector<std::shared_ptr<AttributePairIR>> attributePair;
  
  OperationIR( std::vector<std::string> outputName,
          std::string operationName,
          std::vector<std::string> inputName,
          std::vector<std::shared_ptr<AttributePairIR>> attributePair );
  
  std::shared_ptr<AttributePairIR> getAttributePair( std::string keyName );
  bool hasAttributePair( std::string keyName );
};

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_IR_HPP
