
add_mlir_dialect_library(MLIRGenTen
  IR/GenTenOps.cpp
  ${PROJECT_SOURCE_DIR}/include/Dialect/GenTen/IR/GenTenOps.h
  IR/GenTenPrinter.cpp
  ${PROJECT_SOURCE_DIR}/include/Dialect/GenTen/IR/GenTenPrinter.h
  IR/GenTenCompilerIR.cpp
  ${PROJECT_SOURCE_DIR}/include/Dialect/GenTen/IR/GenTenCompilerIR.hpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GenTen

  DEPENDS
  MLIRGenTenOpsIncGen
  MLIRGenTenStructsIncGen
  MLIRGenTenInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRFunc
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRQuant
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )

add_subdirectory(Transforms)
