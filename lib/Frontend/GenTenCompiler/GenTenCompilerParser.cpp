//===- GenTenCompilerParser.cpp - Parser for the GenTen language -*- C++ -*-===//
//
//===-----------------------------------------------------------------------===//
//
// This file implements a Parser for the GenTen language.
//
// This file was generated.
//
//===-----------------------------------------------------------------------===//

#include "Frontend/GenTenCompiler/GenTenCompilerParser.hpp"

#include <memory>
#include <tuple>

namespace gen_ten_compiler {

class GenTenCompilerParser {
  
public:
  GenTenCompilerParser( const TokenList &tokenList )
 : tokenList( tokenList ), maximumIndex( 0 )
{ }
  
  bool parseLeftBrace( uint32_t &index, std::shared_ptr<LeftBraceAST> &leftBraceAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_leftBrace ) {
      leftBraceAST.reset( new LeftBraceAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseRightBrace( uint32_t &index, std::shared_ptr<RightBraceAST> &rightBraceAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_rightBrace ) {
      rightBraceAST.reset( new RightBraceAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseLeftBracket( uint32_t &index, std::shared_ptr<LeftBracketAST> &leftBracketAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_leftBracket ) {
      leftBracketAST.reset( new LeftBracketAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseRightBracket( uint32_t &index, std::shared_ptr<RightBracketAST> &rightBracketAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_rightBracket ) {
      rightBracketAST.reset( new RightBracketAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseArrow( uint32_t &index, std::shared_ptr<ArrowAST> &arrowAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_arrow ) {
      arrowAST.reset( new ArrowAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseLeftParenthesis( uint32_t &index, std::shared_ptr<LeftParenthesisAST> &leftParenthesisAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_leftParenthesis ) {
      leftParenthesisAST.reset( new LeftParenthesisAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseRightParenthesis( uint32_t &index, std::shared_ptr<RightParenthesisAST> &rightParenthesisAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_rightParenthesis ) {
      rightParenthesisAST.reset( new RightParenthesisAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseLeftAngle( uint32_t &index, std::shared_ptr<LeftAngleAST> &leftAngleAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_leftAngle ) {
      leftAngleAST.reset( new LeftAngleAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseRightAngle( uint32_t &index, std::shared_ptr<RightAngleAST> &rightAngleAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_rightAngle ) {
      rightAngleAST.reset( new RightAngleAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseAsterisk( uint32_t &index, std::shared_ptr<AsteriskAST> &asteriskAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_asterisk ) {
      asteriskAST.reset( new AsteriskAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseComma( uint32_t &index, std::shared_ptr<CommaAST> &commaAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_comma ) {
      commaAST.reset( new CommaAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseColon( uint32_t &index, std::shared_ptr<ColonAST> &colonAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_colon ) {
      colonAST.reset( new ColonAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseEqualSign( uint32_t &index, std::shared_ptr<EqualSignAST> &equalSignAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_equalSign ) {
      equalSignAST.reset( new EqualSignAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parsePeriod( uint32_t &index, std::shared_ptr<PeriodAST> &periodAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_period ) {
      periodAST.reset( new PeriodAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseFuncLiteral( uint32_t &index, std::shared_ptr<FuncLiteralAST> &funcLiteralAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_funcLiteral ) {
      funcLiteralAST.reset( new FuncLiteralAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseVarLiteral( uint32_t &index, std::shared_ptr<VarLiteralAST> &varLiteralAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_varLiteral ) {
      varLiteralAST.reset( new VarLiteralAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseDataTypeFloat32( uint32_t &index, std::shared_ptr<DataTypeFloat32AST> &dataTypeFloat32AST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_dataTypeFloat32 ) {
      dataTypeFloat32AST.reset( new DataTypeFloat32AST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseDataTypeInt64( uint32_t &index, std::shared_ptr<DataTypeInt64AST> &dataTypeInt64AST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_dataTypeInt64 ) {
      dataTypeInt64AST.reset( new DataTypeInt64AST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseQuesionMark( uint32_t &index, std::shared_ptr<QuesionMarkAST> &quesionMarkAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].literalKind == tok_quesionMark ) {
      quesionMarkAST.reset( new QuesionMarkAST() );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseString( uint32_t &index, std::shared_ptr<StringAST> &stringAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].tokenKind == tok_string ) {
      std::string text = tokenList.tokens[ index ].getText();
      stringAST.reset( new StringAST( text ) );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseIdentifier( uint32_t &index, std::shared_ptr<IdentifierAST> &identifierAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].tokenKind == tok_identifier ) {
      std::string text = tokenList.tokens[ index ].getText();
      identifierAST.reset( new IdentifierAST( text ) );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseInteger( uint32_t &index, std::shared_ptr<IntegerAST> &integerAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].tokenKind == tok_integer ) {
      std::string text = tokenList.tokens[ index ].getText();
      integerAST.reset( new IntegerAST( text ) );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseFloatingPoint( uint32_t &index, std::shared_ptr<FloatingPointAST> &floatingPointAST ) {
    if ( index < tokenList.tokens.size() && tokenList.tokens[ index ].tokenKind == tok_floatingPoint ) {
      std::string text = tokenList.tokens[ index ].getText();
      floatingPointAST.reset( new FloatingPointAST( text ) );
      index += 1;
      if ( maximumIndex < index ) {
        maximumIndex = index;
      }
      
      return true;
    }
    return false;
  }
  
  bool parseProgram( uint32_t &index, std::shared_ptr<ProgramAST> &programAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<FunctionAST> functionAST;
    
    if ( !parseFunction( newIndex, functionAST ) ) {
      return false;
    }
    
    if ( newIndex >= tokenList.tokens.size() || tokenList.tokens[ newIndex ].tokenKind != tok_endOfInput ) {
      return false;
    }
    
    programAST.reset( new ProgramAST( functionAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseFunction( uint32_t &index, std::shared_ptr<FunctionAST> &functionAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<FuncLiteralAST> funcLiteralAST;
    std::shared_ptr<FunctionNameAST> functionNameAST;
    std::shared_ptr<InputPhraseAST> inputPhraseAST;
    std::shared_ptr<ArrowAST> arrowAST;
    std::shared_ptr<OutputPhraseAST> outputPhraseAST;
    std::shared_ptr<LeftBraceAST> leftBraceAST;
    std::vector<std::shared_ptr<StatementAST>> statementAST;
    std::shared_ptr<RightBraceAST> rightBraceAST;
    
    if ( !parseFuncLiteral( newIndex, funcLiteralAST ) ) {
      return false;
    }
    
    if ( !parseFunctionName( newIndex, functionNameAST ) ) {
      return false;
    }
    
    if ( !parseInputPhrase( newIndex, inputPhraseAST ) ) {
      return false;
    }
    
    if ( !parseArrow( newIndex, arrowAST ) ) {
      return false;
    }
    
    if ( !parseOutputPhrase( newIndex, outputPhraseAST ) ) {
      return false;
    }
    
    if ( !parseLeftBrace( newIndex, leftBraceAST ) ) {
      return false;
    }
    
    if ( !parseStatement( newIndex, statementAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightBrace( newIndex, rightBraceAST ) ) {
      return false;
    }
    
    functionAST.reset( new FunctionAST( funcLiteralAST, functionNameAST, inputPhraseAST, arrowAST, outputPhraseAST, leftBraceAST, statementAST, rightBraceAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseFunctionName( uint32_t &index, std::shared_ptr<FunctionNameAST> &functionNameAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    functionNameAST.reset( new FunctionNameAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseInputPhrase( uint32_t &index, std::shared_ptr<InputPhraseAST> &inputPhraseAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftParenthesisAST> leftParenthesisAST;
    std::vector<std::shared_ptr<ParameterAST>> parameterAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightParenthesisAST> rightParenthesisAST;
    
    if ( !parseLeftParenthesis( newIndex, leftParenthesisAST ) ) {
      return false;
    }
    
    if ( !parseParameter( newIndex, parameterAST, commaAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightParenthesis( newIndex, rightParenthesisAST ) ) {
      return false;
    }
    
    inputPhraseAST.reset( new InputPhraseAST( leftParenthesisAST, parameterAST, rightParenthesisAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseParameter( uint32_t &index, std::shared_ptr<ParameterAST> &parameterAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<ParameterNameAST> parameterNameAST;
    std::shared_ptr<ColonAST> colonAST;
    std::shared_ptr<TensorAST> tensorAST;
    
    if ( !parseParameterName( newIndex, parameterNameAST ) ) {
      return false;
    }
    
    if ( !parseColon( newIndex, colonAST ) ) {
      return false;
    }
    
    if ( !parseTensor( newIndex, tensorAST ) ) {
      return false;
    }
    
    parameterAST.reset( new ParameterAST( parameterNameAST, colonAST, tensorAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseParameterName( uint32_t &index, std::shared_ptr<ParameterNameAST> &parameterNameAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    parameterNameAST.reset( new ParameterNameAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseOutputPhrase( uint32_t &index, std::shared_ptr<OutputPhraseAST> &outputPhraseAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftParenthesisAST> leftParenthesisAST;
    std::vector<std::shared_ptr<ResultAST>> resultAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightParenthesisAST> rightParenthesisAST;
    
    if ( !parseLeftParenthesis( newIndex, leftParenthesisAST ) ) {
      return false;
    }
    
    if ( !parseResult( newIndex, resultAST, commaAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightParenthesis( newIndex, rightParenthesisAST ) ) {
      return false;
    }
    
    outputPhraseAST.reset( new OutputPhraseAST( leftParenthesisAST, resultAST, rightParenthesisAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseResult( uint32_t &index, std::shared_ptr<ResultAST> &resultAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<ResultNameAST> resultNameAST;
    std::shared_ptr<ColonAST> colonAST;
    std::shared_ptr<TensorAST> tensorAST;
    
    if ( !parseResultName( newIndex, resultNameAST ) ) {
      return false;
    }
    
    if ( !parseColon( newIndex, colonAST ) ) {
      return false;
    }
    
    if ( !parseTensor( newIndex, tensorAST ) ) {
      return false;
    }
    
    resultAST.reset( new ResultAST( resultNameAST, colonAST, tensorAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseResultName( uint32_t &index, std::shared_ptr<ResultNameAST> &resultNameAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    resultNameAST.reset( new ResultNameAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseVariable( uint32_t &index, std::shared_ptr<VariableAST> &variableAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<VarLiteralAST> varLiteralAST;
    std::shared_ptr<VariableNameAST> variableNameAST;
    std::shared_ptr<ColonAST> colonAST;
    std::shared_ptr<TensorAST> tensorAST;
    std::shared_ptr<InitializerAST> initializerAST;
    
    if ( !parseVarLiteral( newIndex, varLiteralAST ) ) {
      return false;
    }
    
    if ( !parseVariableName( newIndex, variableNameAST ) ) {
      return false;
    }
    
    if ( !parseColon( newIndex, colonAST ) ) {
      return false;
    }
    
    if ( !parseTensor( newIndex, tensorAST ) ) {
      return false;
    }
    
    parseInitializer( newIndex, initializerAST );
    
    variableAST.reset( new VariableAST( varLiteralAST, variableNameAST, colonAST, tensorAST, initializerAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseVariableName( uint32_t &index, std::shared_ptr<VariableNameAST> &variableNameAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    variableNameAST.reset( new VariableNameAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseOperation( uint32_t &index, std::shared_ptr<OperationAST> &operationAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<ResultPhraseAST> resultPhraseAST;
    std::shared_ptr<EqualSignAST> equalSignAST;
    std::shared_ptr<OperationNameAST> operationNameAST;
    std::shared_ptr<ArgumentPhraseAST> argumentPhraseAST;
    std::shared_ptr<AttributesAST> attributesAST;
    
    if ( !parseResultPhrase( newIndex, resultPhraseAST ) ) {
      return false;
    }
    
    if ( !parseEqualSign( newIndex, equalSignAST ) ) {
      return false;
    }
    
    if ( !parseOperationName( newIndex, operationNameAST ) ) {
      return false;
    }
    
    if ( !parseArgumentPhrase( newIndex, argumentPhraseAST ) ) {
      return false;
    }
    
    parseAttributes( newIndex, attributesAST );
    
    operationAST.reset( new OperationAST( resultPhraseAST, equalSignAST, operationNameAST, argumentPhraseAST, attributesAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseResultPhrase( uint32_t &index, std::shared_ptr<ResultPhraseAST> &resultPhraseAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftParenthesisAST> leftParenthesisAST;
    std::vector<std::shared_ptr<OutputAST>> outputAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightParenthesisAST> rightParenthesisAST;
    
    if ( !parseLeftParenthesis( newIndex, leftParenthesisAST ) ) {
      return false;
    }
    
    if ( !parseOutput( newIndex, outputAST, commaAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightParenthesis( newIndex, rightParenthesisAST ) ) {
      return false;
    }
    
    resultPhraseAST.reset( new ResultPhraseAST( leftParenthesisAST, outputAST, rightParenthesisAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseOutput( uint32_t &index, std::shared_ptr<OutputAST> &outputAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    outputAST.reset( new OutputAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseOperationName( uint32_t &index, std::shared_ptr<OperationNameAST> &operationNameAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    operationNameAST.reset( new OperationNameAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseArgumentPhrase( uint32_t &index, std::shared_ptr<ArgumentPhraseAST> &argumentPhraseAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftParenthesisAST> leftParenthesisAST;
    std::vector<std::shared_ptr<InputAST>> inputAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightParenthesisAST> rightParenthesisAST;
    
    if ( !parseLeftParenthesis( newIndex, leftParenthesisAST ) ) {
      return false;
    }
    
    if ( !parseInput( newIndex, inputAST, commaAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightParenthesis( newIndex, rightParenthesisAST ) ) {
      return false;
    }
    
    argumentPhraseAST.reset( new ArgumentPhraseAST( leftParenthesisAST, inputAST, rightParenthesisAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseInput( uint32_t &index, std::shared_ptr<InputAST> &inputAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    inputAST.reset( new InputAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseInitializer( uint32_t &index, std::shared_ptr<InitializerAST> &initializerAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<EqualSignAST> equalSignAST;
    std::shared_ptr<TensorValueAST> tensorValueAST;
    
    if ( !parseEqualSign( newIndex, equalSignAST ) ) {
      return false;
    }
    
    if ( !parseTensorValue( newIndex, tensorValueAST ) ) {
      return false;
    }
    
    initializerAST.reset( new InitializerAST( equalSignAST, tensorValueAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseTensorValue( uint32_t &index, std::shared_ptr<TensorValueAST> &tensorValueAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftBracketAST> leftBracketAST;
    std::vector<std::shared_ptr<ConstantValueAST>> constantValueAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightBracketAST> rightBracketAST;
    
    if ( !parseLeftBracket( newIndex, leftBracketAST ) ) {
      return false;
    }
    
    if ( !parseConstantValue( newIndex, constantValueAST, commaAST, 1 ) ) {
      return false;
    }
    
    if ( !parseRightBracket( newIndex, rightBracketAST ) ) {
      return false;
    }
    
    tensorValueAST.reset( new TensorValueAST( leftBracketAST, constantValueAST, rightBracketAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseTensor( uint32_t &index, std::shared_ptr<TensorAST> &tensorAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<DataTypeAST> dataTypeAST;
    std::shared_ptr<LeftBracketAST> leftBracketAST;
    std::vector<std::shared_ptr<DimensionAST>> dimensionAST;
    std::vector<std::shared_ptr<AsteriskAST>> asteriskAST;
    std::shared_ptr<RightBracketAST> rightBracketAST;
    
    if ( !parseDataType( newIndex, dataTypeAST ) ) {
      return false;
    }
    
    if ( !parseLeftBracket( newIndex, leftBracketAST ) ) {
      return false;
    }
    
    if ( !parseDimension( newIndex, dimensionAST, asteriskAST, 0 ) ) {
      return false;
    }
    
    if ( !parseRightBracket( newIndex, rightBracketAST ) ) {
      return false;
    }
    
    tensorAST.reset( new TensorAST( dataTypeAST, leftBracketAST, dimensionAST, rightBracketAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseIntegerDimension( uint32_t &index, std::shared_ptr<IntegerDimensionAST> &integerDimensionAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IntegerAST> integerAST;
    
    if ( !parseInteger( newIndex, integerAST ) ) {
      return false;
    }
    
    integerDimensionAST.reset( new IntegerDimensionAST( integerAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseVariableDimension( uint32_t &index, std::shared_ptr<VariableDimensionAST> &variableDimensionAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    variableDimensionAST.reset( new VariableDimensionAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseUnknownDimension( uint32_t &index, std::shared_ptr<UnknownDimensionAST> &unknownDimensionAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<QuesionMarkAST> quesionMarkAST;
    
    if ( !parseQuesionMark( newIndex, quesionMarkAST ) ) {
      return false;
    }
    
    unknownDimensionAST.reset( new UnknownDimensionAST( quesionMarkAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseAttributes( uint32_t &index, std::shared_ptr<AttributesAST> &attributesAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftBracketAST> leftBracketAST;
    std::vector<std::shared_ptr<AttributePairAST>> attributePairAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightBracketAST> rightBracketAST;
    
    if ( !parseLeftBracket( newIndex, leftBracketAST ) ) {
      return false;
    }
    
    if ( !parseAttributePair( newIndex, attributePairAST, commaAST, 1 ) ) {
      return false;
    }
    
    if ( !parseRightBracket( newIndex, rightBracketAST ) ) {
      return false;
    }
    
    attributesAST.reset( new AttributesAST( leftBracketAST, attributePairAST, rightBracketAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseAttributePair( uint32_t &index, std::shared_ptr<AttributePairAST> &attributePairAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<AttributeKeyAST> attributeKeyAST;
    std::shared_ptr<ColonAST> colonAST;
    std::shared_ptr<AttributeValueAST> attributeValueAST;
    
    if ( !parseAttributeKey( newIndex, attributeKeyAST ) ) {
      return false;
    }
    
    if ( !parseColon( newIndex, colonAST ) ) {
      return false;
    }
    
    if ( !parseAttributeValue( newIndex, attributeValueAST ) ) {
      return false;
    }
    
    attributePairAST.reset( new AttributePairAST( attributeKeyAST, colonAST, attributeValueAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseAttributeKey( uint32_t &index, std::shared_ptr<AttributeKeyAST> &attributeKeyAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<StringAST> stringAST;
    
    if ( !parseString( newIndex, stringAST ) ) {
      return false;
    }
    
    attributeKeyAST.reset( new AttributeKeyAST( stringAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseArrayValue( uint32_t &index, std::shared_ptr<ArrayValueAST> &arrayValueAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<LeftBracketAST> leftBracketAST;
    std::vector<std::shared_ptr<NumberStringValueAST>> numberStringValueAST;
    std::vector<std::shared_ptr<CommaAST>> commaAST;
    std::shared_ptr<RightBracketAST> rightBracketAST;
    
    if ( !parseLeftBracket( newIndex, leftBracketAST ) ) {
      return false;
    }
    
    if ( !parseNumberStringValue( newIndex, numberStringValueAST, commaAST, 1 ) ) {
      return false;
    }
    
    if ( !parseRightBracket( newIndex, rightBracketAST ) ) {
      return false;
    }
    
    arrayValueAST.reset( new ArrayValueAST( leftBracketAST, numberStringValueAST, rightBracketAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseStringValue( uint32_t &index, std::shared_ptr<StringValueAST> &stringValueAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<StringAST> stringAST;
    
    if ( !parseString( newIndex, stringAST ) ) {
      return false;
    }
    
    stringValueAST.reset( new StringValueAST( stringAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseReferenceValue( uint32_t &index, std::shared_ptr<ReferenceValueAST> &referenceValueAST ) {
    uint32_t newIndex = index;
    
    std::shared_ptr<IdentifierAST> identifierAST;
    
    if ( !parseIdentifier( newIndex, identifierAST ) ) {
      return false;
    }
    
    referenceValueAST.reset( new ReferenceValueAST( identifierAST ) );
    index = newIndex;
    return true;
  }
  
  bool parseStatement( uint32_t &index, std::shared_ptr<StatementAST> &statementAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<VariableAST> variableAST;
      if ( parseVariable( newIndex, variableAST ) && ( !matched || newIndex > lastIndex ) ) {
        statementAST.reset( new StatementAST( variableAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<OperationAST> operationAST;
      if ( parseOperation( newIndex, operationAST ) && ( !matched || newIndex > lastIndex ) ) {
        statementAST.reset( new StatementAST( operationAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<FunctionAST> functionAST;
      if ( parseFunction( newIndex, functionAST ) && ( !matched || newIndex > lastIndex ) ) {
        statementAST.reset( new StatementAST( functionAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseConstantValue( uint32_t &index, std::shared_ptr<ConstantValueAST> &constantValueAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<NumberValueAST> numberValueAST;
      if ( parseNumberValue( newIndex, numberValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        constantValueAST.reset( new ConstantValueAST( numberValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<TensorValueAST> tensorValueAST;
      if ( parseTensorValue( newIndex, tensorValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        constantValueAST.reset( new ConstantValueAST( tensorValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<StringValueAST> stringValueAST;
      if ( parseStringValue( newIndex, stringValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        constantValueAST.reset( new ConstantValueAST( stringValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseDataType( uint32_t &index, std::shared_ptr<DataTypeAST> &dataTypeAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<DataTypeFloat32AST> dataTypeFloat32AST;
      if ( parseDataTypeFloat32( newIndex, dataTypeFloat32AST ) && ( !matched || newIndex > lastIndex ) ) {
        dataTypeAST.reset( new DataTypeAST( dataTypeFloat32AST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<DataTypeInt64AST> dataTypeInt64AST;
      if ( parseDataTypeInt64( newIndex, dataTypeInt64AST ) && ( !matched || newIndex > lastIndex ) ) {
        dataTypeAST.reset( new DataTypeAST( dataTypeInt64AST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<QuesionMarkAST> quesionMarkAST;
      if ( parseQuesionMark( newIndex, quesionMarkAST ) && ( !matched || newIndex > lastIndex ) ) {
        dataTypeAST.reset( new DataTypeAST( quesionMarkAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseDimension( uint32_t &index, std::shared_ptr<DimensionAST> &dimensionAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<IntegerDimensionAST> integerDimensionAST;
      if ( parseIntegerDimension( newIndex, integerDimensionAST ) && ( !matched || newIndex > lastIndex ) ) {
        dimensionAST.reset( new DimensionAST( integerDimensionAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<VariableDimensionAST> variableDimensionAST;
      if ( parseVariableDimension( newIndex, variableDimensionAST ) && ( !matched || newIndex > lastIndex ) ) {
        dimensionAST.reset( new DimensionAST( variableDimensionAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<UnknownDimensionAST> unknownDimensionAST;
      if ( parseUnknownDimension( newIndex, unknownDimensionAST ) && ( !matched || newIndex > lastIndex ) ) {
        dimensionAST.reset( new DimensionAST( unknownDimensionAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseAttributeValue( uint32_t &index, std::shared_ptr<AttributeValueAST> &attributeValueAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<ArrayValueAST> arrayValueAST;
      if ( parseArrayValue( newIndex, arrayValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        attributeValueAST.reset( new AttributeValueAST( arrayValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<NumberValueAST> numberValueAST;
      if ( parseNumberValue( newIndex, numberValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        attributeValueAST.reset( new AttributeValueAST( numberValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<StringValueAST> stringValueAST;
      if ( parseStringValue( newIndex, stringValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        attributeValueAST.reset( new AttributeValueAST( stringValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<ReferenceValueAST> referenceValueAST;
      if ( parseReferenceValue( newIndex, referenceValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        attributeValueAST.reset( new AttributeValueAST( referenceValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseNumberStringValue( uint32_t &index, std::shared_ptr<NumberStringValueAST> &numberStringValueAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<NumberValueAST> numberValueAST;
      if ( parseNumberValue( newIndex, numberValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        numberStringValueAST.reset( new NumberStringValueAST( numberValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<ReferenceValueAST> referenceValueAST;
      if ( parseReferenceValue( newIndex, referenceValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        numberStringValueAST.reset( new NumberStringValueAST( referenceValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<StringValueAST> stringValueAST;
      if ( parseStringValue( newIndex, stringValueAST ) && ( !matched || newIndex > lastIndex ) ) {
        numberStringValueAST.reset( new NumberStringValueAST( stringValueAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseNumberValue( uint32_t &index, std::shared_ptr<NumberValueAST> &numberValueAST ) {
    uint32_t lastIndex = index;
    bool matched = false;
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<IntegerAST> integerAST;
      if ( parseInteger( newIndex, integerAST ) && ( !matched || newIndex > lastIndex ) ) {
        numberValueAST.reset( new NumberValueAST( integerAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }
    
    {
      uint32_t newIndex = index;
      std::shared_ptr<FloatingPointAST> floatingPointAST;
      if ( parseFloatingPoint( newIndex, floatingPointAST ) && ( !matched || newIndex > lastIndex ) ) {
        numberValueAST.reset( new NumberValueAST( floatingPointAST ) );
        lastIndex = newIndex;
        matched = true;
      }
    }

    if ( matched ) {
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseStatement( uint32_t &index, std::vector<std::shared_ptr<StatementAST>> &statementAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<StatementAST>> nodes;
    
    uint32_t newIndex = lastIndex;
    std::shared_ptr<StatementAST> node;
    
    while ( parseStatement( newIndex, node ) && newIndex > lastIndex ) {
      lastIndex = newIndex;
      nodes.push_back( node );
    }
    
    if ( nodes.size() >= minimum ) {
      statementAST = std::move( nodes );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseParameter( uint32_t &index, std::vector<std::shared_ptr<ParameterAST>> &parameterAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<ParameterAST>> nodes;
    std::shared_ptr<ParameterAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseParameter( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseParameter( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      parameterAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseResult( uint32_t &index, std::vector<std::shared_ptr<ResultAST>> &resultAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<ResultAST>> nodes;
    std::shared_ptr<ResultAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseResult( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseResult( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      resultAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseOutput( uint32_t &index, std::vector<std::shared_ptr<OutputAST>> &outputAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<OutputAST>> nodes;
    std::shared_ptr<OutputAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseOutput( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseOutput( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      outputAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseInput( uint32_t &index, std::vector<std::shared_ptr<InputAST>> &inputAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<InputAST>> nodes;
    std::shared_ptr<InputAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseInput( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseInput( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      inputAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseConstantValue( uint32_t &index, std::vector<std::shared_ptr<ConstantValueAST>> &constantValueAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<ConstantValueAST>> nodes;
    std::shared_ptr<ConstantValueAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseConstantValue( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseConstantValue( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      constantValueAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseDimension( uint32_t &index, std::vector<std::shared_ptr<DimensionAST>> &dimensionAST, std::vector<std::shared_ptr<AsteriskAST>> &asteriskAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<DimensionAST>> nodes;
    std::shared_ptr<DimensionAST> node;
    std::vector<std::shared_ptr<AsteriskAST>> separators;
    std::shared_ptr<AsteriskAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseDimension( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseAsterisk( newIndex, separator ) && parseDimension( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      dimensionAST = std::move( nodes );
      asteriskAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseAttributePair( uint32_t &index, std::vector<std::shared_ptr<AttributePairAST>> &attributePairAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<AttributePairAST>> nodes;
    std::shared_ptr<AttributePairAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseAttributePair( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseAttributePair( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      attributePairAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  bool parseNumberStringValue( uint32_t &index, std::vector<std::shared_ptr<NumberStringValueAST>> &numberStringValueAST, std::vector<std::shared_ptr<CommaAST>> &commaAST, uint32_t minimum ) {
    uint32_t lastIndex = index;
    
    std::vector<std::shared_ptr<NumberStringValueAST>> nodes;
    std::shared_ptr<NumberStringValueAST> node;
    std::vector<std::shared_ptr<CommaAST>> separators;
    std::shared_ptr<CommaAST> separator;
    
    uint32_t newIndex = lastIndex;
    
    if ( parseNumberStringValue( newIndex, node ) ) {
      nodes.push_back( node );
      
      if ( newIndex > lastIndex ) {
        lastIndex = newIndex;
        
        while ( parseComma( newIndex, separator ) && parseNumberStringValue( newIndex, node ) ) {
          separators.push_back( separator );
          nodes.push_back( node );
          
          if ( newIndex > lastIndex ) {
            lastIndex = newIndex;
          } else {
            break;
          }
        }
      }
    }
    
    if ( nodes.size() >= minimum ) {
      numberStringValueAST = std::move( nodes );
      commaAST = std::move( separators );
      index = lastIndex;
      return true;
    } else {
      return false;
    }
  }
  
  std::shared_ptr<ProgramAST> parseProgram() {
    uint32_t index = 0;
    std::shared_ptr<ProgramAST> programAST;
    parseProgram( index, programAST );
    if ( programAST == nullptr ) {
      const Token &token = tokenList.tokens[ maximumIndex ];
      std::cerr << *token.location.fileName << ":" << token.location.line << ":" << token.location.column << " error: Parser error '" << token.getText() << "'" << std::endl;
    }
    return programAST;
  }
  
private:
  const TokenList &tokenList;
  uint32_t maximumIndex;
};

Parser::Parser( const TokenList &tokenList )
  : tokenList( tokenList )
{ }

std::shared_ptr<ProgramAST> Parser::parseProgram()
{
  GenTenCompilerParser parser( tokenList );
  
  std::shared_ptr<ProgramAST> program = parser.parseProgram();
  
  return program;
}

} // namespace gen_ten_compiler
