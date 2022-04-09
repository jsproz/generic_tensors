//===- GenTenCompilerLexer.h - Lexer for the GenTen language -*- C++ -*-===//
//
//===-----------------------------------------------------------------------===//
//
// This file defines a Lexer interface for the GenTen language.
//
// This file was generated.
//
//===-----------------------------------------------------------------------===//

#ifndef GEN_TEN_COMPILER_GEN_TEN_COMPILER_LEXER_H
#define GEN_TEN_COMPILER_GEN_TEN_COMPILER_LEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace gen_ten_compiler {

struct Location {
  std::shared_ptr<std::string> fileName;
  const char *position;
  size_t length;
  int64_t line;
  int64_t column;
  Location( std::shared_ptr<std::string> fileName, const char *position, size_t length, int64_t line, int64_t column )
    : fileName( fileName ), position( position ), length( length ), line( line ), column( column )
  { }
  Location( std::shared_ptr<std::string> fileName, const char *begin )
    : fileName( fileName ), position( begin ), length( 0 ), line( 0 ), column( 0 )
  { }
};

enum TokenKind : int {
  tok_unknown,
  tok_endOfInput,
  tok_rightBrace,
  tok_leftBrace,
  tok_rightBracket,
  tok_leftBracket,
  tok_dataTypeInt64,
  tok_dataTypeFloat32,
  tok_quesionMark,
  tok_rightAngle,
  tok_equalSign,
  tok_leftAngle,
  tok_colon,
  tok_period,
  tok_arrow,
  tok_comma,
  tok_asterisk,
  tok_rightParenthesis,
  tok_leftParenthesis,
  tok_varLiteral,
  tok_funcLiteral,
  tok_white,
  tok_string,
  tok_identifier,
  tok_integer,
  tok_floatingPoint
};

struct Buffer {
  std::shared_ptr<std::string> fileName;
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer;
  
  Buffer( std::shared_ptr<std::string> fileName, std::unique_ptr<llvm::MemoryBuffer> &memoryBuffer );
  
  static std::shared_ptr<Buffer> buffer( std::shared_ptr<std::string> fileName );
};

struct Token {
  TokenKind tokenKind;
  TokenKind literalKind;
  Location location;
  std::vector<Token> trivia;
  Token( TokenKind tokenKind, TokenKind literalKind, Location location, std::vector<Token> trivia )
    : tokenKind( tokenKind ), literalKind( literalKind ), location( location ), trivia( std::move( trivia ) )
  { }
  Token( TokenKind tokenKind, TokenKind literalKind, Location location )
    : tokenKind( tokenKind ), literalKind( literalKind ), location( location )
  { }
  Token( Location location )
    : tokenKind( tok_unknown ), literalKind( tok_unknown ), location( location )
  { }
  std::string getText() const {
    return std::string( location.position, location.length );
  }
};

std::ostream& operator << ( std::ostream& os, const gen_ten_compiler::Token& token );

struct TokenList {
  std::shared_ptr<Buffer> buffer;
  std::vector<Token> tokens;
  TokenList() { }
  TokenList( std::shared_ptr<Buffer> buffer, std::vector<Token> tokens )
    : buffer( buffer ), tokens( std::move( tokens ) )
  { }
};

class Lexer {
public:
  Lexer( std::shared_ptr<Buffer> buffer );
  TokenList getTokenList();
  static std::shared_ptr<gen_ten_compiler::TokenList> scanInputFile( llvm::StringRef fileName );
private:
  std::shared_ptr<Buffer> buffer;
};

} // namespace gen_ten_compiler

#endif // GEN_TEN_COMPILER_GEN_TEN_COMPILER_LEXER_H
