//
//  GenTenLogger.cpp
//  GenTenCompiler
//
//  Created by Joe Sprowes on 2/15/22.
//

#include "Frontend/GenTenCompiler/GenTenLogger.h"

namespace gen_ten {

Logger::Logger()
: std::ostream( this ), _hasErrors( false )
{ }

Logger &Logger::error() {
  Logger::getInstance()._hasErrors = true;
  Logger::getInstance() << "error: ";
  return Logger::getInstance();
}

Logger &Logger::warning() {
  Logger::getInstance() << "warning: ";
  return Logger::getInstance();
}

Logger &Logger::note() {
  return Logger::getInstance();
}

int Logger::overflow( int c ) {
  print( c );
  return 0;
}

void Logger::print( char c ) {
  std::cout.put( c );
}

Logger &Logger::getInstance() {
  static Logger logger;
  return logger;
}

bool Logger::hasErrors() {
  return Logger::getInstance()._hasErrors;
}

} // namespace gen_ten
