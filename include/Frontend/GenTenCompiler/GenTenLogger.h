//
//  GenTenLogger.hpp
//  GenTenCompiler
//
//  Created by Joe Sprowes on 2/15/22.
//

#ifndef GenTenLogger_h
#define GenTenLogger_h

#include <iostream>

namespace gen_ten {

struct Logger :  private std::streambuf , public std::ostream
{
  Logger();
  static Logger &error();
  static Logger &warning();
  static Logger &note();
  static bool hasErrors();
  
private:
  int overflow( int c ) override;
  void print( char c );
  static Logger &getInstance();
  
  bool _hasErrors;
};

} // namespace gen_ten

#endif // GenTenLogger_h
