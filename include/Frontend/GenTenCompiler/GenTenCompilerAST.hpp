//===- GenTenCompilerAST.h - AST for the GenTen language -*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file defines a Abstract Syntax Tree interface for the GenTen language.
//
// This file was generated.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_AST_H
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_AST_H

#include <memory>
#include <string>
#include <vector>

namespace gen_ten_compiler {

enum ASTNodeKind {
  ast_unknown,
  ast_leftBrace,
  ast_rightBrace,
  ast_leftBracket,
  ast_rightBracket,
  ast_arrow,
  ast_leftParenthesis,
  ast_rightParenthesis,
  ast_leftAngle,
  ast_rightAngle,
  ast_asterisk,
  ast_comma,
  ast_colon,
  ast_equalSign,
  ast_period,
  ast_funcLiteral,
  ast_varLiteral,
  ast_dataTypeFloat32,
  ast_dataTypeInt64,
  ast_quesionMark,
  ast_string,
  ast_identifier,
  ast_integer,
  ast_floatingPoint,
  ast_program,
  ast_function,
  ast_functionName,
  ast_inputPhrase,
  ast_parameter,
  ast_parameterName,
  ast_outputPhrase,
  ast_result,
  ast_resultName,
  ast_variable,
  ast_variableName,
  ast_operation,
  ast_resultPhrase,
  ast_output,
  ast_operationName,
  ast_argumentPhrase,
  ast_input,
  ast_initializer,
  ast_tensorValue,
  ast_tensor,
  ast_integerDimension,
  ast_variableDimension,
  ast_unknownDimension,
  ast_attributes,
  ast_attributePair,
  ast_attributeKey,
  ast_arrayValue,
  ast_stringValue,
  ast_referenceValue,
  ast_statement,
  ast_constantValue,
  ast_dataType,
  ast_dimension,
  ast_attributeValue,
  ast_numberStringValue,
  ast_numberValue
};

struct ASTNode {
  ASTNodeKind kind;
  ASTNode( ASTNodeKind kind )
    : kind( kind )
  { }
};

struct LeftBraceAST;
struct RightBraceAST;
struct LeftBracketAST;
struct RightBracketAST;
struct ArrowAST;
struct LeftParenthesisAST;
struct RightParenthesisAST;
struct LeftAngleAST;
struct RightAngleAST;
struct AsteriskAST;
struct CommaAST;
struct ColonAST;
struct EqualSignAST;
struct PeriodAST;
struct FuncLiteralAST;
struct VarLiteralAST;
struct DataTypeFloat32AST;
struct DataTypeInt64AST;
struct QuesionMarkAST;
struct StringAST;
struct IdentifierAST;
struct IntegerAST;
struct FloatingPointAST;
struct ProgramAST;
struct FunctionAST;
struct FunctionNameAST;
struct InputPhraseAST;
struct ParameterAST;
struct ParameterNameAST;
struct OutputPhraseAST;
struct ResultAST;
struct ResultNameAST;
struct VariableAST;
struct VariableNameAST;
struct OperationAST;
struct ResultPhraseAST;
struct OutputAST;
struct OperationNameAST;
struct ArgumentPhraseAST;
struct InputAST;
struct InitializerAST;
struct TensorValueAST;
struct TensorAST;
struct IntegerDimensionAST;
struct VariableDimensionAST;
struct UnknownDimensionAST;
struct AttributesAST;
struct AttributePairAST;
struct AttributeKeyAST;
struct ArrayValueAST;
struct StringValueAST;
struct ReferenceValueAST;
struct StatementAST;
struct ConstantValueAST;
struct DataTypeAST;
struct DimensionAST;
struct AttributeValueAST;
struct NumberStringValueAST;
struct NumberValueAST;


struct LeftBraceAST : public ASTNode {
  LeftBraceAST();
};

struct RightBraceAST : public ASTNode {
  RightBraceAST();
};

struct LeftBracketAST : public ASTNode {
  LeftBracketAST();
};

struct RightBracketAST : public ASTNode {
  RightBracketAST();
};

struct ArrowAST : public ASTNode {
  ArrowAST();
};

struct LeftParenthesisAST : public ASTNode {
  LeftParenthesisAST();
};

struct RightParenthesisAST : public ASTNode {
  RightParenthesisAST();
};

struct LeftAngleAST : public ASTNode {
  LeftAngleAST();
};

struct RightAngleAST : public ASTNode {
  RightAngleAST();
};

struct AsteriskAST : public ASTNode {
  AsteriskAST();
};

struct CommaAST : public ASTNode {
  CommaAST();
};

struct ColonAST : public ASTNode {
  ColonAST();
};

struct EqualSignAST : public ASTNode {
  EqualSignAST();
};

struct PeriodAST : public ASTNode {
  PeriodAST();
};

struct FuncLiteralAST : public ASTNode {
  FuncLiteralAST();
};

struct VarLiteralAST : public ASTNode {
  VarLiteralAST();
};

struct DataTypeFloat32AST : public ASTNode {
  DataTypeFloat32AST();
};

struct DataTypeInt64AST : public ASTNode {
  DataTypeInt64AST();
};

struct QuesionMarkAST : public ASTNode {
  QuesionMarkAST();
};

struct StringAST : public ASTNode {
  std::string text;
  
  StringAST( std::string text );
};

struct IdentifierAST : public ASTNode {
  std::string text;
  
  IdentifierAST( std::string text );
};

struct IntegerAST : public ASTNode {
  std::string text;
  
  IntegerAST( std::string text );
};

struct FloatingPointAST : public ASTNode {
  std::string text;
  
  FloatingPointAST( std::string text );
};

struct ProgramAST : public ASTNode {
  std::shared_ptr<FunctionAST> function;
  
  ProgramAST( std::shared_ptr<FunctionAST> function );
};

struct FunctionAST : public ASTNode {
  std::shared_ptr<FuncLiteralAST> funcLiteral;
  std::shared_ptr<FunctionNameAST> functionName;
  std::shared_ptr<InputPhraseAST> inputPhrase;
  std::shared_ptr<ArrowAST> arrow;
  std::shared_ptr<OutputPhraseAST> outputPhrase;
  std::shared_ptr<LeftBraceAST> leftBrace;
  std::vector<std::shared_ptr<StatementAST>> statement;
  std::shared_ptr<RightBraceAST> rightBrace;
  
  FunctionAST( std::shared_ptr<FuncLiteralAST> funcLiteral, std::shared_ptr<FunctionNameAST> functionName, std::shared_ptr<InputPhraseAST> inputPhrase, std::shared_ptr<ArrowAST> arrow, std::shared_ptr<OutputPhraseAST> outputPhrase, std::shared_ptr<LeftBraceAST> leftBrace, std::vector<std::shared_ptr<StatementAST>> statement, std::shared_ptr<RightBraceAST> rightBrace );
};

struct FunctionNameAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  FunctionNameAST( std::shared_ptr<IdentifierAST> identifier );
};

struct InputPhraseAST : public ASTNode {
  std::shared_ptr<LeftParenthesisAST> leftParenthesis;
  std::vector<std::shared_ptr<ParameterAST>> parameter;
  std::shared_ptr<RightParenthesisAST> rightParenthesis;
  
  InputPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<ParameterAST>> parameter, std::shared_ptr<RightParenthesisAST> rightParenthesis );
};

struct ParameterAST : public ASTNode {
  std::shared_ptr<ParameterNameAST> parameterName;
  std::shared_ptr<ColonAST> colon;
  std::shared_ptr<TensorAST> tensor;
  
  ParameterAST( std::shared_ptr<ParameterNameAST> parameterName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor );
};

struct ParameterNameAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  ParameterNameAST( std::shared_ptr<IdentifierAST> identifier );
};

struct OutputPhraseAST : public ASTNode {
  std::shared_ptr<LeftParenthesisAST> leftParenthesis;
  std::vector<std::shared_ptr<ResultAST>> result;
  std::shared_ptr<RightParenthesisAST> rightParenthesis;
  
  OutputPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<ResultAST>> result, std::shared_ptr<RightParenthesisAST> rightParenthesis );
};

struct ResultAST : public ASTNode {
  std::shared_ptr<ResultNameAST> resultName;
  std::shared_ptr<ColonAST> colon;
  std::shared_ptr<TensorAST> tensor;
  
  ResultAST( std::shared_ptr<ResultNameAST> resultName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor );
};

struct ResultNameAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  ResultNameAST( std::shared_ptr<IdentifierAST> identifier );
};

struct VariableAST : public ASTNode {
  std::shared_ptr<VarLiteralAST> varLiteral;
  std::shared_ptr<VariableNameAST> variableName;
  std::shared_ptr<ColonAST> colon;
  std::shared_ptr<TensorAST> tensor;
  std::shared_ptr<InitializerAST> initializer;
  
  VariableAST( std::shared_ptr<VarLiteralAST> varLiteral, std::shared_ptr<VariableNameAST> variableName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor, std::shared_ptr<InitializerAST> initializer );
};

struct VariableNameAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  VariableNameAST( std::shared_ptr<IdentifierAST> identifier );
};

struct OperationAST : public ASTNode {
  std::shared_ptr<ResultPhraseAST> resultPhrase;
  std::shared_ptr<EqualSignAST> equalSign;
  std::shared_ptr<OperationNameAST> operationName;
  std::shared_ptr<ArgumentPhraseAST> argumentPhrase;
  std::shared_ptr<AttributesAST> attributes;
  
  OperationAST( std::shared_ptr<ResultPhraseAST> resultPhrase, std::shared_ptr<EqualSignAST> equalSign, std::shared_ptr<OperationNameAST> operationName, std::shared_ptr<ArgumentPhraseAST> argumentPhrase, std::shared_ptr<AttributesAST> attributes );
};

struct ResultPhraseAST : public ASTNode {
  std::shared_ptr<LeftParenthesisAST> leftParenthesis;
  std::vector<std::shared_ptr<OutputAST>> output;
  std::shared_ptr<RightParenthesisAST> rightParenthesis;
  
  ResultPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<OutputAST>> output, std::shared_ptr<RightParenthesisAST> rightParenthesis );
};

struct OutputAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  OutputAST( std::shared_ptr<IdentifierAST> identifier );
};

struct OperationNameAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  OperationNameAST( std::shared_ptr<IdentifierAST> identifier );
};

struct ArgumentPhraseAST : public ASTNode {
  std::shared_ptr<LeftParenthesisAST> leftParenthesis;
  std::vector<std::shared_ptr<InputAST>> input;
  std::shared_ptr<RightParenthesisAST> rightParenthesis;
  
  ArgumentPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<InputAST>> input, std::shared_ptr<RightParenthesisAST> rightParenthesis );
};

struct InputAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  InputAST( std::shared_ptr<IdentifierAST> identifier );
};

struct InitializerAST : public ASTNode {
  std::shared_ptr<EqualSignAST> equalSign;
  std::shared_ptr<TensorValueAST> tensorValue;
  
  InitializerAST( std::shared_ptr<EqualSignAST> equalSign, std::shared_ptr<TensorValueAST> tensorValue );
};

struct TensorValueAST : public ASTNode {
  std::shared_ptr<LeftBracketAST> leftBracket;
  std::vector<std::shared_ptr<ConstantValueAST>> constantValue;
  std::shared_ptr<RightBracketAST> rightBracket;
  
  TensorValueAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<ConstantValueAST>> constantValue, std::shared_ptr<RightBracketAST> rightBracket );
};

struct TensorAST : public ASTNode {
  std::shared_ptr<DataTypeAST> dataType;
  std::shared_ptr<LeftBracketAST> leftBracket;
  std::vector<std::shared_ptr<DimensionAST>> dimension;
  std::shared_ptr<RightBracketAST> rightBracket;
  
  TensorAST( std::shared_ptr<DataTypeAST> dataType, std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<DimensionAST>> dimension, std::shared_ptr<RightBracketAST> rightBracket );
};

struct IntegerDimensionAST : public ASTNode {
  std::shared_ptr<IntegerAST> integer;
  
  IntegerDimensionAST( std::shared_ptr<IntegerAST> integer );
};

struct VariableDimensionAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  VariableDimensionAST( std::shared_ptr<IdentifierAST> identifier );
};

struct UnknownDimensionAST : public ASTNode {
  std::shared_ptr<QuesionMarkAST> quesionMark;
  
  UnknownDimensionAST( std::shared_ptr<QuesionMarkAST> quesionMark );
};

struct AttributesAST : public ASTNode {
  std::shared_ptr<LeftBracketAST> leftBracket;
  std::vector<std::shared_ptr<AttributePairAST>> attributePair;
  std::shared_ptr<RightBracketAST> rightBracket;
  
  AttributesAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<AttributePairAST>> attributePair, std::shared_ptr<RightBracketAST> rightBracket );
};

struct AttributePairAST : public ASTNode {
  std::shared_ptr<AttributeKeyAST> attributeKey;
  std::shared_ptr<ColonAST> colon;
  std::shared_ptr<AttributeValueAST> attributeValue;
  
  AttributePairAST( std::shared_ptr<AttributeKeyAST> attributeKey, std::shared_ptr<ColonAST> colon, std::shared_ptr<AttributeValueAST> attributeValue );
};

struct AttributeKeyAST : public ASTNode {
  std::shared_ptr<StringAST> string;
  
  AttributeKeyAST( std::shared_ptr<StringAST> string );
};

struct ArrayValueAST : public ASTNode {
  std::shared_ptr<LeftBracketAST> leftBracket;
  std::vector<std::shared_ptr<NumberStringValueAST>> numberStringValue;
  std::shared_ptr<RightBracketAST> rightBracket;
  
  ArrayValueAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<NumberStringValueAST>> numberStringValue, std::shared_ptr<RightBracketAST> rightBracket );
};

struct StringValueAST : public ASTNode {
  std::shared_ptr<StringAST> string;
  
  StringValueAST( std::shared_ptr<StringAST> string );
};

struct ReferenceValueAST : public ASTNode {
  std::shared_ptr<IdentifierAST> identifier;
  
  ReferenceValueAST( std::shared_ptr<IdentifierAST> identifier );
};

struct StatementAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  StatementAST();
  StatementAST( std::shared_ptr<VariableAST> variable );
  StatementAST( std::shared_ptr<OperationAST> operation );
  StatementAST( std::shared_ptr<FunctionAST> function );
  std::shared_ptr<VariableAST> getVariableAST();
  std::shared_ptr<OperationAST> getOperationAST();
  std::shared_ptr<FunctionAST> getFunctionAST();
};

struct ConstantValueAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  ConstantValueAST();
  ConstantValueAST( std::shared_ptr<NumberValueAST> numberValue );
  ConstantValueAST( std::shared_ptr<TensorValueAST> tensorValue );
  ConstantValueAST( std::shared_ptr<StringValueAST> stringValue );
  std::shared_ptr<NumberValueAST> getNumberValueAST();
  std::shared_ptr<TensorValueAST> getTensorValueAST();
  std::shared_ptr<StringValueAST> getStringValueAST();
};

struct DataTypeAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  DataTypeAST();
  DataTypeAST( std::shared_ptr<DataTypeFloat32AST> dataTypeFloat32 );
  DataTypeAST( std::shared_ptr<DataTypeInt64AST> dataTypeInt64 );
  DataTypeAST( std::shared_ptr<QuesionMarkAST> quesionMark );
  std::shared_ptr<DataTypeFloat32AST> getDataTypeFloat32AST();
  std::shared_ptr<DataTypeInt64AST> getDataTypeInt64AST();
  std::shared_ptr<QuesionMarkAST> getQuesionMarkAST();
};

struct DimensionAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  DimensionAST();
  DimensionAST( std::shared_ptr<IntegerDimensionAST> integerDimension );
  DimensionAST( std::shared_ptr<VariableDimensionAST> variableDimension );
  DimensionAST( std::shared_ptr<UnknownDimensionAST> unknownDimension );
  std::shared_ptr<IntegerDimensionAST> getIntegerDimensionAST();
  std::shared_ptr<VariableDimensionAST> getVariableDimensionAST();
  std::shared_ptr<UnknownDimensionAST> getUnknownDimensionAST();
};

struct AttributeValueAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  AttributeValueAST();
  AttributeValueAST( std::shared_ptr<ArrayValueAST> arrayValue );
  AttributeValueAST( std::shared_ptr<NumberValueAST> numberValue );
  AttributeValueAST( std::shared_ptr<StringValueAST> stringValue );
  AttributeValueAST( std::shared_ptr<ReferenceValueAST> referenceValue );
  std::shared_ptr<ArrayValueAST> getArrayValueAST();
  std::shared_ptr<NumberValueAST> getNumberValueAST();
  std::shared_ptr<StringValueAST> getStringValueAST();
  std::shared_ptr<ReferenceValueAST> getReferenceValueAST();
};

struct NumberStringValueAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  NumberStringValueAST();
  NumberStringValueAST( std::shared_ptr<NumberValueAST> numberValue );
  NumberStringValueAST( std::shared_ptr<ReferenceValueAST> referenceValue );
  NumberStringValueAST( std::shared_ptr<StringValueAST> stringValue );
  std::shared_ptr<NumberValueAST> getNumberValueAST();
  std::shared_ptr<ReferenceValueAST> getReferenceValueAST();
  std::shared_ptr<StringValueAST> getStringValueAST();
};

struct NumberValueAST : public ASTNode {
  ASTNodeKind childKind;
  std::shared_ptr<ASTNode> child;
  
  NumberValueAST();
  NumberValueAST( std::shared_ptr<IntegerAST> integer );
  NumberValueAST( std::shared_ptr<FloatingPointAST> floatingPoint );
  std::shared_ptr<IntegerAST> getIntegerAST();
  std::shared_ptr<FloatingPointAST> getFloatingPointAST();
};

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_AST_H
