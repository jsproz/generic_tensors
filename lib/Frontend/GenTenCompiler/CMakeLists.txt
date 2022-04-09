
add_mlir_dialect_library(GenTenCompiler
        GenTenCompilerAST.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenCompilerAST.hpp
        GenTenCompilerGenTenGen.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenCompilerGenTenGen.h
        GenTenCompilerLexer.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenCompilerLexer.hpp
        GenTenCompilerParser.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenCompilerParser.hpp
        GenTenCompilerMlirGen.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenCompilerMlirGen.h
        GenTenLogger.cpp
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler/GenTenLogger.h
        
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Frontend/GenTenCompiler

	LINK_LIBS PUBLIC
	MLIRIR
	)
