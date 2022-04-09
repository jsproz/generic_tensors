set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(generic-tensors
  generic_tensors.cpp

  DEPENDS
  GenTenCompiler
  )
llvm_update_compile_flags(generic-tensors)
target_link_libraries(generic-tensors
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRCallInterfaces
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  GenTenCompiler
  )

mlir_check_link_libraries(generic-tensors)
