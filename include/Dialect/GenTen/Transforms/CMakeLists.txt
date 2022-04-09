set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GenTenOpt)
add_public_tablegen_target(MLIRGenTenPassIncGen)
add_dependencies(mlir-headers MLIRGenTenPassIncGen)

add_mlir_doc(Passes GenTenPasses ./ -gen-pass-doc)
