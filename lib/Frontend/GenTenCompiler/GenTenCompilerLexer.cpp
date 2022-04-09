//===- GenTenCompilerLexer.cpp - Lexer for the GenTen language -*- C++ -*-===//
//
//===-----------------------------------------------------------------------===//
//
// This file implements a Lexer for the GenTen language.
//
// This file was generated.
//
//===-----------------------------------------------------------------------===//

#include "Frontend/GenTenCompiler/GenTenCompilerLexer.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

namespace gen_ten_compiler {

std::ostream& operator << ( std::ostream& os, const gen_ten_compiler::Token& token )
{
  os << *token.location.fileName << ":"  << token.location.line << ":" << token.location.column << " (" << token.literalKind << "," << token.tokenKind << ") " << token.getText();
  return os;
}

Buffer::Buffer( std::shared_ptr<std::string> fileName, std::unique_ptr<llvm::MemoryBuffer> &memoryBuffer )
: fileName( fileName ), memoryBuffer( std::move( memoryBuffer ) )
{ }

std::shared_ptr<Buffer> Buffer::buffer( std::shared_ptr<std::string> fileName )
{
  std::shared_ptr<Buffer> buffer;
  
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrError = llvm::MemoryBuffer::getFileOrSTDIN( *fileName );
  if ( bufferOrError ) {
    buffer.reset( new Buffer( fileName, bufferOrError.get() ) );
  }
  
  return buffer;
}

struct Utf8Buffer {
  enum Utf8BufferState {
    waitingForFirstByte,
    waitingForLastByte,
    waitingForNextToLastByte,
    waitingForSecondToLastByte,
    finished
  };
  std::shared_ptr<std::string> fileName;
  const char *begin, *start, *current, *next, *end;
  int64_t line;
  int64_t column;
  bool eolState;
  Utf8Buffer( std::shared_ptr<std::string> fileName, const char *begin, const char *end )
    : fileName( fileName ), begin( begin ), start( begin ), current( begin ), next( begin ), end( end ), line( 1 ), column( 0 ), eolState( false )
  { }
  Utf8Buffer( )
    : fileName( nullptr ), begin( nullptr ), start( nullptr ), current( nullptr ), next( nullptr ), end( nullptr ), line( 1 ), column( 0 ), eolState( false )
  { }
  bool isAtEnd() {
    return current >= end;
  }
  size_t bytesAvailable() {
    return size_t( end - current );
  }
  uint8_t getNextByte() {
    uint8_t nextByte = *current;
    current += 1;
    return nextByte;
  }
  void update() {
    const char *originalCurrent = current;
    current = start;
    
    while ( current < originalCurrent ) {
      char32_t character = getNextCharacter();
      
      if ( !eolState ) {
        switch ( character ) {
          case 0x0D:
            eolState = true;
            break;
          case 0x0A:
          case 0x2028:
          case 0x2029:
            line += 1;
            column = 1;
            break;
          default:
            column += 1;
        }
      } else {
        switch ( character ) {
          case 0x0D:
          case 0x2028:
          case 0x2029:
            line += 1;
            column = 1;
            break;
          case 0x0A:
            break;
          default:
            eolState = false;
        }
      }
    }
    start = current;
  }
  Token getToken( TokenKind tokenKind, TokenKind literalKind ) {
    Location location( fileName, start, current - start, line, column );
    Token token( tokenKind, literalKind, location );
    update();
    return token;
  }
  bool isFirstAndOnlyByte( uint8_t byte ) {
    return ( byte & 0x80 ) == 0x00;
  }
  bool isFirstOfTwoByte( uint8_t byte ) {
    return ( byte & 0xE0 ) == 0xC0;
  }
  bool isFirstOfThreeByte( uint8_t byte ) {
    return ( byte & 0xF0 ) == 0xE0;
  }
  bool isFirstOfFourByte( uint8_t byte ) {
    return ( byte & 0xF8 ) == 0xF0;
  }
  bool isNextByte( uint8_t byte ) {
    return ( byte & 0xC0 ) == 0x80;
  }
  void consumeCharacter() {
    current = next;
  }
  char32_t getNextCharacter() {
    char32_t character = 0xFFFD;
    uint32_t unicodeScalarValue = 0x0;
    Utf8BufferState state = waitingForFirstByte;
    
    while ( !isAtEnd() && state != finished ) {
      uint32_t nextByte = getNextByte();
      switch ( state ) {
        case waitingForFirstByte:
          if ( isFirstAndOnlyByte( nextByte ) ) {
            unicodeScalarValue = nextByte;
            character = unicodeScalarValue;
            state = finished;
          } else if ( isFirstOfTwoByte( nextByte ) ) {
            unicodeScalarValue = ( nextByte & 0x1F );
            state = waitingForLastByte;
          } else if ( isFirstOfThreeByte( nextByte ) ) {
            unicodeScalarValue = ( nextByte & 0x0F );
            state = waitingForNextToLastByte;
          } else if ( isFirstOfFourByte( nextByte ) ) {
            unicodeScalarValue = ( nextByte & 0x07 );
            state = waitingForSecondToLastByte;
          } else {
            while ( isNextByte( nextByte ) && !isAtEnd() ) {
              getNextByte();
            }
            character = 0xFFFD;
          }
          break;
        case waitingForLastByte:
          if ( isNextByte( nextByte ) && !isAtEnd() ) {
            unicodeScalarValue = ( unicodeScalarValue << 6 ) | ( nextByte & 0x3F );
            character = unicodeScalarValue;
            state = finished;
          } else {
            while ( isNextByte( nextByte ) && !isAtEnd() ) {
              getNextByte();
            }
            character = 0xFFFD;
          }
          break;
        case waitingForNextToLastByte:
          if ( isNextByte( nextByte ) && !isAtEnd() ) {
            unicodeScalarValue = ( unicodeScalarValue << 6 ) | ( nextByte & 0x3F );
            state = waitingForLastByte;
          } else {
            while ( isNextByte( nextByte ) && !isAtEnd() ) {
              getNextByte();
            }
            character = 0xFFFD;
          }
          break;
        case waitingForSecondToLastByte:
          if ( isNextByte( nextByte ) && !isAtEnd() ) {
            unicodeScalarValue = ( unicodeScalarValue << 6 ) | ( nextByte & 0x3F );
            state = waitingForNextToLastByte;
          } else {
            while ( isNextByte( nextByte ) && !isAtEnd() ) {
              getNextByte();
            }
            character = 0xFFFD;
          }
          break;
        case finished:
          break;
      }
    }
    
    next = current;
    
    return character;
  }
};

class GenTenCompilerLexer {
public:
  GenTenCompilerLexer( std::shared_ptr<Buffer> buffer )
    : buffer( buffer )
  {
    llvm::StringRef stringRef( buffer->memoryBuffer->getBuffer() );
    utf8Buffer =  Utf8Buffer( buffer->fileName, stringRef.begin(), stringRef.end() );
  }
  
  TokenList getTokenList() {
    std::vector<Token> tokens;
    
    while ( utf8Buffer.current < utf8Buffer.end ) {
      Token token = getNextToken();
      tokens.push_back( token );
    }
    
    TokenList tokenList( buffer, tokens );
    return tokenList;
  }
  
private:
  bool matchesLiteral( const char *literal, size_t length ) {
    if ( utf8Buffer.bytesAvailable() < length ) {
      return false;
    }
    
    size_t index = 0;
    while ( index < length ) {
      if ( *( literal + index ) != *( utf8Buffer.current + index ) ) {
        return false;
      }
      index += 1;
    }
    
    return true;
  }
  
  void matchLiteral() {
    literalKind = tok_unknown;
    
    if ( matchesLiteral( "}", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_rightBrace;
      return;
    } else if ( matchesLiteral( "{", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_leftBrace;
      return;
    } else if ( matchesLiteral( "]", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_rightBracket;
      return;
    } else if ( matchesLiteral( "[", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_leftBracket;
      return;
    } else if ( matchesLiteral( "Int64", 5 )  ) {
      utf8Buffer.current += 5;
      literalKind = tok_dataTypeInt64;
      return;
    } else if ( matchesLiteral( "Float32", 7 )  ) {
      utf8Buffer.current += 7;
      literalKind = tok_dataTypeFloat32;
      return;
    } else if ( matchesLiteral( "\?", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_quesionMark;
      return;
    } else if ( matchesLiteral( ">", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_rightAngle;
      return;
    } else if ( matchesLiteral( "=", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_equalSign;
      return;
    } else if ( matchesLiteral( "<", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_leftAngle;
      return;
    } else if ( matchesLiteral( ":", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_colon;
      return;
    } else if ( matchesLiteral( ".", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_period;
      return;
    } else if ( matchesLiteral( "->", 2 )  ) {
      utf8Buffer.current += 2;
      literalKind = tok_arrow;
      return;
    } else if ( matchesLiteral( ",", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_comma;
      return;
    } else if ( matchesLiteral( "*", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_asterisk;
      return;
    } else if ( matchesLiteral( ")", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_rightParenthesis;
      return;
    } else if ( matchesLiteral( "(", 1 )  ) {
      utf8Buffer.current += 1;
      literalKind = tok_leftParenthesis;
      return;
    } else if ( matchesLiteral( "%var", 4 )  ) {
      utf8Buffer.current += 4;
      literalKind = tok_varLiteral;
      return;
    } else if ( matchesLiteral( "%func", 5 )  ) {
      utf8Buffer.current += 5;
      literalKind = tok_funcLiteral;
      return;
    }
  }
  
  // U+0009,U+0020
  bool scanSet1() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 9 || unicodeScalarValue == 32 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0009,U+0020
  bool scanNotSet1() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet1() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0009,U+0020
  bool scanSet1( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet1() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0009,U+0020
  bool scanSet1( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet1() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000D
  bool scanSet2() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 13 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+000D
  bool scanNotSet2() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet2() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+000D
  bool scanSet2( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet2() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000D
  bool scanSet2( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet2() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A
  bool scanSet3() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 10 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+000A
  bool scanNotSet3() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet3() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+000A
  bool scanSet3( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet3() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A
  bool scanSet3( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet3() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002F
  bool scanSet4() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 47 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+002F
  bool scanNotSet4() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet4() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+002F
  bool scanSet4( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet4() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002F
  bool scanSet4( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet4() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A,U+000D
  bool scanSet5() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 10 || unicodeScalarValue == 13 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+000A,U+000D
  bool scanNotSet5() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet5() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+000A,U+000D
  bool scanSet5( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet5() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A,U+000D
  bool scanSet5( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet5() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002A
  bool scanSet6() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 42 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+002A
  bool scanNotSet6() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet6() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+002A
  bool scanSet6( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet6() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002A
  bool scanSet6( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet6() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0022
  bool scanSet7() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 34 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0022
  bool scanNotSet7() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet7() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0022
  bool scanSet7( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet7() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0022
  bool scanSet7( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet7() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A,U+000D,U+0022,U+005C
  bool scanSet8() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 10 || unicodeScalarValue == 13 || unicodeScalarValue == 34 || unicodeScalarValue == 92 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+000A,U+000D,U+0022,U+005C
  bool scanNotSet8() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet8() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+000A,U+000D,U+0022,U+005C
  bool scanSet8( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet8() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+000A,U+000D,U+0022,U+005C
  bool scanSet8( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet8() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+005C
  bool scanSet9() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 92 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+005C
  bool scanNotSet9() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet9() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+005C
  bool scanSet9( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet9() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+005C
  bool scanSet9( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet9() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0075
  bool scanSet10() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 117 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0075
  bool scanNotSet10() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet10() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0075
  bool scanSet10( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet10() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0075
  bool scanSet10( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet10() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+007B
  bool scanSet11() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 123 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+007B
  bool scanNotSet11() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet11() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+007B
  bool scanSet11( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet11() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+007B
  bool scanSet11( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet11() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+007D
  bool scanSet12() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 125 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+007D
  bool scanNotSet12() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet12() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+007D
  bool scanSet12( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet12() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+007D
  bool scanSet12( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet12() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039,U+0041-U+0046,U+0061-U+0066
  bool scanSet13() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( ( unicodeScalarValue <= 57 && unicodeScalarValue >= 48 ) || ( unicodeScalarValue <= 70 && unicodeScalarValue >= 65 ) || ( unicodeScalarValue <= 102 && unicodeScalarValue >= 97 ) ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0030-U+0039,U+0041-U+0046,U+0061-U+0066
  bool scanNotSet13() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet13() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0030-U+0039,U+0041-U+0046,U+0061-U+0066
  bool scanSet13( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet13() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039,U+0041-U+0046,U+0061-U+0066
  bool scanSet13( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet13() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0022,U+0027,U+0030,U+005C,U+006E,U+0072,U+0074
  bool scanSet14() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 34 || unicodeScalarValue == 39 || unicodeScalarValue == 48 || unicodeScalarValue == 92 || unicodeScalarValue == 110 || unicodeScalarValue == 114 || unicodeScalarValue == 116 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0022,U+0027,U+0030,U+005C,U+006E,U+0072,U+0074
  bool scanNotSet14() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet14() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0022,U+0027,U+0030,U+005C,U+006E,U+0072,U+0074
  bool scanSet14( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet14() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0022,U+0027,U+0030,U+005C,U+006E,U+0072,U+0074
  bool scanSet14( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet14() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet15() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( ( unicodeScalarValue <= 90 && unicodeScalarValue >= 65 ) || unicodeScalarValue == 95 || ( unicodeScalarValue <= 122 && unicodeScalarValue >= 97 ) ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanNotSet15() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet15() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet15( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet15() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet15( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet15() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039,U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet16() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( ( unicodeScalarValue <= 57 && unicodeScalarValue >= 48 ) || ( unicodeScalarValue <= 90 && unicodeScalarValue >= 65 ) || unicodeScalarValue == 95 || ( unicodeScalarValue <= 122 && unicodeScalarValue >= 97 ) ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0030-U+0039,U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanNotSet16() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet16() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0030-U+0039,U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet16( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet16() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039,U+0041-U+005A,U+005F,U+0061-U+007A
  bool scanSet16( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet16() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002D
  bool scanSet17() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 45 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+002D
  bool scanNotSet17() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet17() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+002D
  bool scanSet17( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet17() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002D
  bool scanSet17( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet17() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030
  bool scanSet18() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 48 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0030
  bool scanNotSet18() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet18() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0030
  bool scanSet18( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet18() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030
  bool scanSet18( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet18() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0031-U+0039
  bool scanSet19() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( ( unicodeScalarValue <= 57 && unicodeScalarValue >= 49 ) ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0031-U+0039
  bool scanNotSet19() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet19() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0031-U+0039
  bool scanSet19( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet19() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0031-U+0039
  bool scanSet19( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet19() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002E
  bool scanSet20() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 46 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+002E
  bool scanNotSet20() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet20() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+002E
  bool scanSet20( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet20() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002E
  bool scanSet20( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet20() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0045,U+0065
  bool scanSet21() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 69 || unicodeScalarValue == 101 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0045,U+0065
  bool scanNotSet21() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet21() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0045,U+0065
  bool scanSet21( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet21() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0045,U+0065
  bool scanSet21( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet21() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002B,U+002D
  bool scanSet22() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( unicodeScalarValue == 43 || unicodeScalarValue == 45 ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+002B,U+002D
  bool scanNotSet22() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet22() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+002B,U+002D
  bool scanSet22( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet22() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+002B,U+002D
  bool scanSet22( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet22() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039
  bool scanSet23() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    char32_t unicodeScalarValue = utf8Buffer.getNextCharacter();
    if ( ( unicodeScalarValue <= 57 && unicodeScalarValue >= 48 ) ) {
      utf8Buffer.consumeCharacter();
      return true;
    }
    utf8Buffer.current = position;
    return false;
  }
  
  // ~U+0030-U+0039
  bool scanNotSet23() {
    if ( utf8Buffer.isAtEnd() ) {
      return false;
    }
    const char *position = utf8Buffer.current;
    if ( scanSet23() ) {
      utf8Buffer.current = position;
      return false;
    } else {
      utf8Buffer.consumeCharacter();
      return true;
    }
  }
  
  // U+0030-U+0039
  bool scanSet23( int64_t minimum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( scanSet23() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  // U+0030-U+0039
  bool scanSet23( int64_t minimum, int64_t maximum ) {
    const char *position = utf8Buffer.current;
    int64_t count = 0;
    while ( count < maximum && scanSet23() ) {
      count += 1;
    }
    if ( count < minimum ) {
      utf8Buffer.current = position;
      return false;
    }
    return true;
  }
  
  bool scanWhiteSpace() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanWhiteCharacter();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanMultipleLineComment();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanSingleLineComment();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotWhiteSpace() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanWhiteSpace();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanWhiteSpace( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanWhiteSpace() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanWhiteSpace( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanWhiteSpace() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanWhiteCharacter() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet1();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanNewLine();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotWhiteCharacter() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanWhiteCharacter();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanWhiteCharacter( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanWhiteCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanWhiteCharacter( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanWhiteCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNewLine() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet2() && scanSet3();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanSet2();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanSet3();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotNewLine() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanNewLine();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanNewLine( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanNewLine() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNewLine( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanNewLine() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanSingleLineComment() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet4() && scanSet4() && scanSingleLineCommentElement( 0 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotSingleLineComment() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanSingleLineComment();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanSingleLineComment( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanSingleLineComment() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanSingleLineComment( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanSingleLineComment() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanSingleLineCommentElement() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanNotSet5();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotSingleLineCommentElement() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanSingleLineCommentElement();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanSingleLineCommentElement( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanSingleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanSingleLineCommentElement( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanSingleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineComment() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet4() && scanSet6() && scanMultipleLineCommentElement( 0 ) && scanMultipleLineCommentEnd();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotMultipleLineComment() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanMultipleLineComment();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanMultipleLineComment( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanMultipleLineComment() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineComment( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanMultipleLineComment() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineCommentElement() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanNotNonmultipleLineCommentElement();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanMultipleLineComment();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotMultipleLineCommentElement() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanMultipleLineCommentElement();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanMultipleLineCommentElement( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanMultipleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineCommentElement( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanMultipleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNonmultipleLineCommentElement() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanMultipleLineCommentEnd();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanMultipleLineComment();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotNonmultipleLineCommentElement() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanNonmultipleLineCommentElement();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanNonmultipleLineCommentElement( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanNonmultipleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNonmultipleLineCommentElement( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanNonmultipleLineCommentElement() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineCommentEnd() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet6() && scanSet4();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotMultipleLineCommentEnd() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanMultipleLineCommentEnd();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanMultipleLineCommentEnd( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanMultipleLineCommentEnd() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanMultipleLineCommentEnd( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanMultipleLineCommentEnd() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanQuotedTextItem() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanQuotedTextItemCharacter();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanEscapedCharacter();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotQuotedTextItem() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanQuotedTextItem();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanQuotedTextItem( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanQuotedTextItem() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanQuotedTextItem( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanQuotedTextItem() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanQuotedTextItemCharacter() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanNotNonquotedTextItemCharacter();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotQuotedTextItemCharacter() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanQuotedTextItemCharacter();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanQuotedTextItemCharacter( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanQuotedTextItemCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanQuotedTextItemCharacter( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanQuotedTextItemCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNonquotedTextItemCharacter() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet8();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotNonquotedTextItemCharacter() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanNonquotedTextItemCharacter();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanNonquotedTextItemCharacter( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanNonquotedTextItemCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanNonquotedTextItemCharacter( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanNonquotedTextItemCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedCharacter() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanEscapedUnicodeScalar();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanEscapedCCharacter();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotEscapedCharacter() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanEscapedCharacter();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanEscapedCharacter( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanEscapedCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedCharacter( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanEscapedCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedUnicodeScalar() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet9() && scanSet10() && scanSet11() && scanUnicodeScalarDigit( 1, 8 ) && scanSet12();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotEscapedUnicodeScalar() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanEscapedUnicodeScalar();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanEscapedUnicodeScalar( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanEscapedUnicodeScalar() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedUnicodeScalar( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanEscapedUnicodeScalar() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanUnicodeScalarDigit() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet13();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotUnicodeScalarDigit() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanUnicodeScalarDigit();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanUnicodeScalarDigit( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanUnicodeScalarDigit() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanUnicodeScalarDigit( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanUnicodeScalarDigit() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedCCharacter() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet9() && scanSet14();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotEscapedCCharacter() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanEscapedCCharacter();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanEscapedCCharacter( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanEscapedCCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanEscapedCCharacter( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanEscapedCCharacter() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanIdentifierFollower() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet16();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotIdentifierFollower() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanIdentifierFollower();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanIdentifierFollower( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanIdentifierFollower() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanIdentifierFollower( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanIdentifierFollower() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanFractionalConstant() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanDigitSequence( 0, 1 ) && scanSet20() && scanDigitSequence();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanDigitSequence() && scanSet20();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotFractionalConstant() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanFractionalConstant();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanFractionalConstant( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanFractionalConstant() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanFractionalConstant( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanFractionalConstant() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanExponentPart() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet21() && scanSet22( 0, 1 ) && scanDigitSequence();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotExponentPart() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanExponentPart();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanExponentPart( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanExponentPart() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanExponentPart( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanExponentPart() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanDigitSequence() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanDigit( 1 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotDigitSequence() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanDigitSequence();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanDigitSequence( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanDigitSequence() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanDigitSequence( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanDigitSequence() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanDigit() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet23();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotDigit() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanDigit();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanDigit( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanDigit() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanDigit( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanDigit() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanWhite() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanWhiteSpace( 1 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotWhite() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanWhite();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanWhite( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanWhite() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanWhite( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanWhite() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanString() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet7() && scanQuotedTextItem( 0 ) && scanSet7();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotString() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanString();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanString( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanString() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanString( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanString() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanIdentifier() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet15() && scanIdentifierFollower( 0 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotIdentifier() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanIdentifier();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanIdentifier( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanIdentifier() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanIdentifier( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanIdentifier() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanInteger() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet17( 0, 1 ) && scanSet18();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanSet17( 0, 1 ) && scanSet19() && scanDigit( 0 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotInteger() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanInteger();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanInteger( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanInteger() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanInteger( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanInteger() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanFloatingPoint() {
    const char *originalCurrent = utf8Buffer.current;
    bool matches = false;
    if ( !matches ) {
      matches = scanSet17( 0, 1 ) && scanFractionalConstant() && scanExponentPart( 0, 1 );
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    if ( !matches ) {
      matches = scanSet17( 0, 1 ) && scanDigitSequence() && scanExponentPart();
    }
    if ( !matches ) {
      utf8Buffer.current = originalCurrent;
    }
    return matches;
  }
  
  bool scanNotFloatingPoint() {
    if ( utf8Buffer.current >= utf8Buffer.end ) {
      return false;
    }
    const char *originalCurrent = utf8Buffer.current;
    bool matches = scanFloatingPoint();
    if ( matches ) {
      utf8Buffer.current = originalCurrent;
      matches = false;
    } else {
      utf8Buffer.consumeCharacter();
      matches = true;
    }
    return matches;
  }
  
  bool scanFloatingPoint( int32_t minimum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( scanFloatingPoint() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  bool scanFloatingPoint( int32_t minimum, int32_t maximum ) {
    const char *originalCurrent = utf8Buffer.current;
    const char *current = utf8Buffer.current;
    int32_t count = 0;
    while ( count < maximum && scanFloatingPoint() ) {
      count += 1;
      if ( count >= minimum && utf8Buffer.current == current ) {
        break;
      }
      current = utf8Buffer.current;
    }
    if ( count < minimum ) {
      utf8Buffer.current = originalCurrent;
      return false;
    }
    return true;
  }
  
  void matchTrivia() {
    tokenKind = tok_unknown;
    const char *originalCurrent = utf8Buffer.start;
    const char *lastEnd = utf8Buffer.start;
    
    utf8Buffer.current = originalCurrent;
    if ( scanWhite() && utf8Buffer.current > lastEnd ) {
      tokenKind = tok_white;
      lastEnd = utf8Buffer.current;
    }
    
    utf8Buffer.current = lastEnd;
  }
  
  void getTrivia() {
    trivia.clear();
    
    while ( utf8Buffer.current < utf8Buffer.end ) {
      matchTrivia();
      if ( tokenKind == tok_unknown ) {
        break;
      }
      
      Token token = utf8Buffer.getToken( tokenKind, tok_unknown );
      trivia.push_back( token );
    }
  }
  
  void matchToken() {
    tokenKind = tok_unknown;
    const char *originalCurrent = utf8Buffer.start;
    const char *lastEnd = utf8Buffer.start;
    
    utf8Buffer.current = originalCurrent;
    if ( scanString() && utf8Buffer.current > lastEnd ) {
      tokenKind = tok_string;
      lastEnd = utf8Buffer.current;
    }
    
    utf8Buffer.current = originalCurrent;
    if ( scanIdentifier() && utf8Buffer.current > lastEnd ) {
      tokenKind = tok_identifier;
      lastEnd = utf8Buffer.current;
    }
    
    utf8Buffer.current = originalCurrent;
    if ( scanInteger() && utf8Buffer.current > lastEnd ) {
      tokenKind = tok_integer;
      lastEnd = utf8Buffer.current;
    }
    
    utf8Buffer.current = originalCurrent;
    if ( scanFloatingPoint() && utf8Buffer.current > lastEnd ) {
      tokenKind = tok_floatingPoint;
      lastEnd = utf8Buffer.current;
    }
    
    utf8Buffer.current = lastEnd;
  }
  
  void matchAnyToken() {
    matchTrivia();
    if ( tokenKind != tok_unknown ) {
      return;
    }
    
    matchLiteral();
    if ( literalKind != tok_unknown ) {
      return;
    }
    
    matchToken();
  }
  
  void matchUnknown() {
    const char *originalStart = utf8Buffer.start;
    
    while ( utf8Buffer.start < utf8Buffer.end ) {
      matchAnyToken();
      
      if ( tokenKind != tok_unknown || literalKind != tok_unknown ) {
        break;
      }
      
      utf8Buffer.start += 1;
    }
    
    literalKind = tok_unknown;
    tokenKind = tok_unknown;
    utf8Buffer.start = originalStart;
  }
  
  Token getNextToken() {
    getTrivia();
    
    matchLiteral();
    
    size_t literalLength = ( literalKind == tok_unknown ) ? 0 : size_t( utf8Buffer.current - utf8Buffer.start );
    
    matchToken();
    
    size_t tokenLength = ( tokenKind == tok_unknown ) ? 0 : size_t( utf8Buffer.current - utf8Buffer.start );
    
    if ( literalKind == tok_unknown && tokenKind == tok_unknown ) {
      matchUnknown();
    }
    
    if ( literalLength < tokenLength ) {
      literalKind = tok_unknown;
    }
    
    if ( tokenLength < literalLength ) {
      tokenKind = tok_unknown;
    }
    
    if ( literalKind != tok_unknown || tokenKind != tok_unknown ) {
      utf8Buffer.current = utf8Buffer.start + std::max( literalLength, tokenLength );
    } else if ( utf8Buffer.bytesAvailable() == 0 ) {
      tokenKind = tok_endOfInput;
    }
    
    Token token = utf8Buffer.getToken( tokenKind, literalKind );
    return token;
  }
  
  Utf8Buffer utf8Buffer;
  std::shared_ptr<Buffer> buffer;
  
  std::vector<Token> trivia;
  
  TokenKind literalKind;
  TokenKind tokenKind;
};

Lexer::Lexer( std::shared_ptr<Buffer> buffer )
  : buffer( buffer )
{ }

TokenList Lexer::getTokenList()
{
  GenTenCompilerLexer lexer( buffer );
  TokenList tokenList = lexer.getTokenList();
  return tokenList;
}

std::shared_ptr<gen_ten_compiler::TokenList> Lexer::scanInputFile( llvm::StringRef fileName )
{
  std::shared_ptr<gen_ten_compiler::Buffer> buffer = gen_ten_compiler::Buffer::buffer( std::make_shared<std::string>( fileName ) );
  if ( buffer == nullptr ) {
    return nullptr;
  }
  
  gen_ten_compiler::Lexer lexer( buffer );
  gen_ten_compiler::TokenList tokenList = lexer.getTokenList();
  return std::make_shared<gen_ten_compiler::TokenList>( tokenList );
}

} // namespace gen_ten_compiler
