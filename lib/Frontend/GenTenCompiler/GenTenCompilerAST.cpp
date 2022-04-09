//===- GenTenCompilerAST.cpp - AST for the GenTen language -*- C++ -*-===//
//
//===-----------------------------------------------------------------------===//
//
// This file implements a Abstract Syntax Tree for the GenTen language.
//
// This file was generated.
//
//===-----------------------------------------------------------------------===//

#include "Frontend/GenTenCompiler/GenTenCompilerAST.hpp"

namespace gen_ten_compiler {

LeftBraceAST::LeftBraceAST()
  : ASTNode( ast_leftBrace )
{ }

RightBraceAST::RightBraceAST()
  : ASTNode( ast_rightBrace )
{ }

LeftBracketAST::LeftBracketAST()
  : ASTNode( ast_leftBracket )
{ }

RightBracketAST::RightBracketAST()
  : ASTNode( ast_rightBracket )
{ }

ArrowAST::ArrowAST()
  : ASTNode( ast_arrow )
{ }

LeftParenthesisAST::LeftParenthesisAST()
  : ASTNode( ast_leftParenthesis )
{ }

RightParenthesisAST::RightParenthesisAST()
  : ASTNode( ast_rightParenthesis )
{ }

LeftAngleAST::LeftAngleAST()
  : ASTNode( ast_leftAngle )
{ }

RightAngleAST::RightAngleAST()
  : ASTNode( ast_rightAngle )
{ }

AsteriskAST::AsteriskAST()
  : ASTNode( ast_asterisk )
{ }

CommaAST::CommaAST()
  : ASTNode( ast_comma )
{ }

ColonAST::ColonAST()
  : ASTNode( ast_colon )
{ }

EqualSignAST::EqualSignAST()
  : ASTNode( ast_equalSign )
{ }

PeriodAST::PeriodAST()
  : ASTNode( ast_period )
{ }

FuncLiteralAST::FuncLiteralAST()
  : ASTNode( ast_funcLiteral )
{ }

VarLiteralAST::VarLiteralAST()
  : ASTNode( ast_varLiteral )
{ }

DataTypeFloat32AST::DataTypeFloat32AST()
  : ASTNode( ast_dataTypeFloat32 )
{ }

DataTypeInt64AST::DataTypeInt64AST()
  : ASTNode( ast_dataTypeInt64 )
{ }

QuesionMarkAST::QuesionMarkAST()
  : ASTNode( ast_quesionMark )
{ }

StringAST::StringAST( std::string text )
  : ASTNode( ast_string ), text( std::move( text ) )
{ }

IdentifierAST::IdentifierAST( std::string text )
  : ASTNode( ast_identifier ), text( std::move( text ) )
{ }

IntegerAST::IntegerAST( std::string text )
  : ASTNode( ast_integer ), text( std::move( text ) )
{ }

FloatingPointAST::FloatingPointAST( std::string text )
  : ASTNode( ast_floatingPoint ), text( std::move( text ) )
{ }

ProgramAST::ProgramAST( std::shared_ptr<FunctionAST> function )
  : ASTNode( ast_program ), function( function )
{ }

FunctionAST::FunctionAST( std::shared_ptr<FuncLiteralAST> funcLiteral, std::shared_ptr<FunctionNameAST> functionName, std::shared_ptr<InputPhraseAST> inputPhrase, std::shared_ptr<ArrowAST> arrow, std::shared_ptr<OutputPhraseAST> outputPhrase, std::shared_ptr<LeftBraceAST> leftBrace, std::vector<std::shared_ptr<StatementAST>> statement, std::shared_ptr<RightBraceAST> rightBrace )
  : ASTNode( ast_function ), funcLiteral( funcLiteral ), functionName( functionName ), inputPhrase( inputPhrase ), arrow( arrow ), outputPhrase( outputPhrase ), leftBrace( leftBrace ), statement( std::move( statement ) ), rightBrace( rightBrace )
{ }

FunctionNameAST::FunctionNameAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_functionName ), identifier( identifier )
{ }

InputPhraseAST::InputPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<ParameterAST>> parameter, std::shared_ptr<RightParenthesisAST> rightParenthesis )
  : ASTNode( ast_inputPhrase ), leftParenthesis( leftParenthesis ), parameter( std::move( parameter ) ), rightParenthesis( rightParenthesis )
{ }

ParameterAST::ParameterAST( std::shared_ptr<ParameterNameAST> parameterName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor )
  : ASTNode( ast_parameter ), parameterName( parameterName ), colon( colon ), tensor( tensor )
{ }

ParameterNameAST::ParameterNameAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_parameterName ), identifier( identifier )
{ }

OutputPhraseAST::OutputPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<ResultAST>> result, std::shared_ptr<RightParenthesisAST> rightParenthesis )
  : ASTNode( ast_outputPhrase ), leftParenthesis( leftParenthesis ), result( std::move( result ) ), rightParenthesis( rightParenthesis )
{ }

ResultAST::ResultAST( std::shared_ptr<ResultNameAST> resultName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor )
  : ASTNode( ast_result ), resultName( resultName ), colon( colon ), tensor( tensor )
{ }

ResultNameAST::ResultNameAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_resultName ), identifier( identifier )
{ }

VariableAST::VariableAST( std::shared_ptr<VarLiteralAST> varLiteral, std::shared_ptr<VariableNameAST> variableName, std::shared_ptr<ColonAST> colon, std::shared_ptr<TensorAST> tensor, std::shared_ptr<InitializerAST> initializer )
  : ASTNode( ast_variable ), varLiteral( varLiteral ), variableName( variableName ), colon( colon ), tensor( tensor ), initializer( initializer )
{ }

VariableNameAST::VariableNameAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_variableName ), identifier( identifier )
{ }

OperationAST::OperationAST( std::shared_ptr<ResultPhraseAST> resultPhrase, std::shared_ptr<EqualSignAST> equalSign, std::shared_ptr<OperationNameAST> operationName, std::shared_ptr<ArgumentPhraseAST> argumentPhrase, std::shared_ptr<AttributesAST> attributes )
  : ASTNode( ast_operation ), resultPhrase( resultPhrase ), equalSign( equalSign ), operationName( operationName ), argumentPhrase( argumentPhrase ), attributes( attributes )
{ }

ResultPhraseAST::ResultPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<OutputAST>> output, std::shared_ptr<RightParenthesisAST> rightParenthesis )
  : ASTNode( ast_resultPhrase ), leftParenthesis( leftParenthesis ), output( std::move( output ) ), rightParenthesis( rightParenthesis )
{ }

OutputAST::OutputAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_output ), identifier( identifier )
{ }

OperationNameAST::OperationNameAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_operationName ), identifier( identifier )
{ }

ArgumentPhraseAST::ArgumentPhraseAST( std::shared_ptr<LeftParenthesisAST> leftParenthesis, std::vector<std::shared_ptr<InputAST>> input, std::shared_ptr<RightParenthesisAST> rightParenthesis )
  : ASTNode( ast_argumentPhrase ), leftParenthesis( leftParenthesis ), input( std::move( input ) ), rightParenthesis( rightParenthesis )
{ }

InputAST::InputAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_input ), identifier( identifier )
{ }

InitializerAST::InitializerAST( std::shared_ptr<EqualSignAST> equalSign, std::shared_ptr<TensorValueAST> tensorValue )
  : ASTNode( ast_initializer ), equalSign( equalSign ), tensorValue( tensorValue )
{ }

TensorValueAST::TensorValueAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<ConstantValueAST>> constantValue, std::shared_ptr<RightBracketAST> rightBracket )
  : ASTNode( ast_tensorValue ), leftBracket( leftBracket ), constantValue( std::move( constantValue ) ), rightBracket( rightBracket )
{ }

TensorAST::TensorAST( std::shared_ptr<DataTypeAST> dataType, std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<DimensionAST>> dimension, std::shared_ptr<RightBracketAST> rightBracket )
  : ASTNode( ast_tensor ), dataType( dataType ), leftBracket( leftBracket ), dimension( std::move( dimension ) ), rightBracket( rightBracket )
{ }

IntegerDimensionAST::IntegerDimensionAST( std::shared_ptr<IntegerAST> integer )
  : ASTNode( ast_integerDimension ), integer( integer )
{ }

VariableDimensionAST::VariableDimensionAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_variableDimension ), identifier( identifier )
{ }

UnknownDimensionAST::UnknownDimensionAST( std::shared_ptr<QuesionMarkAST> quesionMark )
  : ASTNode( ast_unknownDimension ), quesionMark( quesionMark )
{ }

AttributesAST::AttributesAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<AttributePairAST>> attributePair, std::shared_ptr<RightBracketAST> rightBracket )
  : ASTNode( ast_attributes ), leftBracket( leftBracket ), attributePair( std::move( attributePair ) ), rightBracket( rightBracket )
{ }

AttributePairAST::AttributePairAST( std::shared_ptr<AttributeKeyAST> attributeKey, std::shared_ptr<ColonAST> colon, std::shared_ptr<AttributeValueAST> attributeValue )
  : ASTNode( ast_attributePair ), attributeKey( attributeKey ), colon( colon ), attributeValue( attributeValue )
{ }

AttributeKeyAST::AttributeKeyAST( std::shared_ptr<StringAST> string )
  : ASTNode( ast_attributeKey ), string( string )
{ }

ArrayValueAST::ArrayValueAST( std::shared_ptr<LeftBracketAST> leftBracket, std::vector<std::shared_ptr<NumberStringValueAST>> numberStringValue, std::shared_ptr<RightBracketAST> rightBracket )
  : ASTNode( ast_arrayValue ), leftBracket( leftBracket ), numberStringValue( std::move( numberStringValue ) ), rightBracket( rightBracket )
{ }

StringValueAST::StringValueAST( std::shared_ptr<StringAST> string )
  : ASTNode( ast_stringValue ), string( string )
{ }

ReferenceValueAST::ReferenceValueAST( std::shared_ptr<IdentifierAST> identifier )
  : ASTNode( ast_referenceValue ), identifier( identifier )
{ }

StatementAST::StatementAST() : ASTNode( ast_statement ), childKind( ast_unknown ) { }
StatementAST::StatementAST( std::shared_ptr<VariableAST> variable ) : ASTNode( ast_statement ), childKind( ast_variable ), child( variable ) { }
StatementAST::StatementAST( std::shared_ptr<OperationAST> operation ) : ASTNode( ast_statement ), childKind( ast_operation ), child( operation ) { }
StatementAST::StatementAST( std::shared_ptr<FunctionAST> function ) : ASTNode( ast_statement ), childKind( ast_function ), child( function ) { }
std::shared_ptr<VariableAST> StatementAST::getVariableAST() { return ( childKind == ast_variable ) ? std::static_pointer_cast<VariableAST>(child) : std::shared_ptr<VariableAST>(); }
std::shared_ptr<OperationAST> StatementAST::getOperationAST() { return ( childKind == ast_operation ) ? std::static_pointer_cast<OperationAST>(child) : std::shared_ptr<OperationAST>(); }
std::shared_ptr<FunctionAST> StatementAST::getFunctionAST() { return ( childKind == ast_function ) ? std::static_pointer_cast<FunctionAST>(child) : std::shared_ptr<FunctionAST>(); }

ConstantValueAST::ConstantValueAST() : ASTNode( ast_constantValue ), childKind( ast_unknown ) { }
ConstantValueAST::ConstantValueAST( std::shared_ptr<NumberValueAST> numberValue ) : ASTNode( ast_constantValue ), childKind( ast_numberValue ), child( numberValue ) { }
ConstantValueAST::ConstantValueAST( std::shared_ptr<TensorValueAST> tensorValue ) : ASTNode( ast_constantValue ), childKind( ast_tensorValue ), child( tensorValue ) { }
ConstantValueAST::ConstantValueAST( std::shared_ptr<StringValueAST> stringValue ) : ASTNode( ast_constantValue ), childKind( ast_stringValue ), child( stringValue ) { }
std::shared_ptr<NumberValueAST> ConstantValueAST::getNumberValueAST() { return ( childKind == ast_numberValue ) ? std::static_pointer_cast<NumberValueAST>(child) : std::shared_ptr<NumberValueAST>(); }
std::shared_ptr<TensorValueAST> ConstantValueAST::getTensorValueAST() { return ( childKind == ast_tensorValue ) ? std::static_pointer_cast<TensorValueAST>(child) : std::shared_ptr<TensorValueAST>(); }
std::shared_ptr<StringValueAST> ConstantValueAST::getStringValueAST() { return ( childKind == ast_stringValue ) ? std::static_pointer_cast<StringValueAST>(child) : std::shared_ptr<StringValueAST>(); }

DataTypeAST::DataTypeAST() : ASTNode( ast_dataType ), childKind( ast_unknown ) { }
DataTypeAST::DataTypeAST( std::shared_ptr<DataTypeFloat32AST> dataTypeFloat32 ) : ASTNode( ast_dataType ), childKind( ast_dataTypeFloat32 ), child( dataTypeFloat32 ) { }
DataTypeAST::DataTypeAST( std::shared_ptr<DataTypeInt64AST> dataTypeInt64 ) : ASTNode( ast_dataType ), childKind( ast_dataTypeInt64 ), child( dataTypeInt64 ) { }
DataTypeAST::DataTypeAST( std::shared_ptr<QuesionMarkAST> quesionMark ) : ASTNode( ast_dataType ), childKind( ast_quesionMark ), child( quesionMark ) { }
std::shared_ptr<DataTypeFloat32AST> DataTypeAST::getDataTypeFloat32AST() { return ( childKind == ast_dataTypeFloat32 ) ? std::static_pointer_cast<DataTypeFloat32AST>(child) : std::shared_ptr<DataTypeFloat32AST>(); }
std::shared_ptr<DataTypeInt64AST> DataTypeAST::getDataTypeInt64AST() { return ( childKind == ast_dataTypeInt64 ) ? std::static_pointer_cast<DataTypeInt64AST>(child) : std::shared_ptr<DataTypeInt64AST>(); }
std::shared_ptr<QuesionMarkAST> DataTypeAST::getQuesionMarkAST() { return ( childKind == ast_quesionMark ) ? std::static_pointer_cast<QuesionMarkAST>(child) : std::shared_ptr<QuesionMarkAST>(); }

DimensionAST::DimensionAST() : ASTNode( ast_dimension ), childKind( ast_unknown ) { }
DimensionAST::DimensionAST( std::shared_ptr<IntegerDimensionAST> integerDimension ) : ASTNode( ast_dimension ), childKind( ast_integerDimension ), child( integerDimension ) { }
DimensionAST::DimensionAST( std::shared_ptr<VariableDimensionAST> variableDimension ) : ASTNode( ast_dimension ), childKind( ast_variableDimension ), child( variableDimension ) { }
DimensionAST::DimensionAST( std::shared_ptr<UnknownDimensionAST> unknownDimension ) : ASTNode( ast_dimension ), childKind( ast_unknownDimension ), child( unknownDimension ) { }
std::shared_ptr<IntegerDimensionAST> DimensionAST::getIntegerDimensionAST() { return ( childKind == ast_integerDimension ) ? std::static_pointer_cast<IntegerDimensionAST>(child) : std::shared_ptr<IntegerDimensionAST>(); }
std::shared_ptr<VariableDimensionAST> DimensionAST::getVariableDimensionAST() { return ( childKind == ast_variableDimension ) ? std::static_pointer_cast<VariableDimensionAST>(child) : std::shared_ptr<VariableDimensionAST>(); }
std::shared_ptr<UnknownDimensionAST> DimensionAST::getUnknownDimensionAST() { return ( childKind == ast_unknownDimension ) ? std::static_pointer_cast<UnknownDimensionAST>(child) : std::shared_ptr<UnknownDimensionAST>(); }

AttributeValueAST::AttributeValueAST() : ASTNode( ast_attributeValue ), childKind( ast_unknown ) { }
AttributeValueAST::AttributeValueAST( std::shared_ptr<ArrayValueAST> arrayValue ) : ASTNode( ast_attributeValue ), childKind( ast_arrayValue ), child( arrayValue ) { }
AttributeValueAST::AttributeValueAST( std::shared_ptr<NumberValueAST> numberValue ) : ASTNode( ast_attributeValue ), childKind( ast_numberValue ), child( numberValue ) { }
AttributeValueAST::AttributeValueAST( std::shared_ptr<StringValueAST> stringValue ) : ASTNode( ast_attributeValue ), childKind( ast_stringValue ), child( stringValue ) { }
AttributeValueAST::AttributeValueAST( std::shared_ptr<ReferenceValueAST> referenceValue ) : ASTNode( ast_attributeValue ), childKind( ast_referenceValue ), child( referenceValue ) { }
std::shared_ptr<ArrayValueAST> AttributeValueAST::getArrayValueAST() { return ( childKind == ast_arrayValue ) ? std::static_pointer_cast<ArrayValueAST>(child) : std::shared_ptr<ArrayValueAST>(); }
std::shared_ptr<NumberValueAST> AttributeValueAST::getNumberValueAST() { return ( childKind == ast_numberValue ) ? std::static_pointer_cast<NumberValueAST>(child) : std::shared_ptr<NumberValueAST>(); }
std::shared_ptr<StringValueAST> AttributeValueAST::getStringValueAST() { return ( childKind == ast_stringValue ) ? std::static_pointer_cast<StringValueAST>(child) : std::shared_ptr<StringValueAST>(); }
std::shared_ptr<ReferenceValueAST> AttributeValueAST::getReferenceValueAST() { return ( childKind == ast_referenceValue ) ? std::static_pointer_cast<ReferenceValueAST>(child) : std::shared_ptr<ReferenceValueAST>(); }

NumberStringValueAST::NumberStringValueAST() : ASTNode( ast_numberStringValue ), childKind( ast_unknown ) { }
NumberStringValueAST::NumberStringValueAST( std::shared_ptr<NumberValueAST> numberValue ) : ASTNode( ast_numberStringValue ), childKind( ast_numberValue ), child( numberValue ) { }
NumberStringValueAST::NumberStringValueAST( std::shared_ptr<ReferenceValueAST> referenceValue ) : ASTNode( ast_numberStringValue ), childKind( ast_referenceValue ), child( referenceValue ) { }
NumberStringValueAST::NumberStringValueAST( std::shared_ptr<StringValueAST> stringValue ) : ASTNode( ast_numberStringValue ), childKind( ast_stringValue ), child( stringValue ) { }
std::shared_ptr<NumberValueAST> NumberStringValueAST::getNumberValueAST() { return ( childKind == ast_numberValue ) ? std::static_pointer_cast<NumberValueAST>(child) : std::shared_ptr<NumberValueAST>(); }
std::shared_ptr<ReferenceValueAST> NumberStringValueAST::getReferenceValueAST() { return ( childKind == ast_referenceValue ) ? std::static_pointer_cast<ReferenceValueAST>(child) : std::shared_ptr<ReferenceValueAST>(); }
std::shared_ptr<StringValueAST> NumberStringValueAST::getStringValueAST() { return ( childKind == ast_stringValue ) ? std::static_pointer_cast<StringValueAST>(child) : std::shared_ptr<StringValueAST>(); }

NumberValueAST::NumberValueAST() : ASTNode( ast_numberValue ), childKind( ast_unknown ) { }
NumberValueAST::NumberValueAST( std::shared_ptr<IntegerAST> integer ) : ASTNode( ast_numberValue ), childKind( ast_integer ), child( integer ) { }
NumberValueAST::NumberValueAST( std::shared_ptr<FloatingPointAST> floatingPoint ) : ASTNode( ast_numberValue ), childKind( ast_floatingPoint ), child( floatingPoint ) { }
std::shared_ptr<IntegerAST> NumberValueAST::getIntegerAST() { return ( childKind == ast_integer ) ? std::static_pointer_cast<IntegerAST>(child) : std::shared_ptr<IntegerAST>(); }
std::shared_ptr<FloatingPointAST> NumberValueAST::getFloatingPointAST() { return ( childKind == ast_floatingPoint ) ? std::static_pointer_cast<FloatingPointAST>(child) : std::shared_ptr<FloatingPointAST>(); }

} // namespace gen_ten_compiler
