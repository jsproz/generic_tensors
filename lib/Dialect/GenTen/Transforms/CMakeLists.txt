add_mlir_dialect_library(MLIRGenTenTransforms
  GenTenCleanUp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GenTen/Transforms

  DEPENDS
  MLIRGenTenPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRGenTen
  )
