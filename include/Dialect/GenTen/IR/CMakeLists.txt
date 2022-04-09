add_mlir_dialect(GenTenOps gen_ten)
add_mlir_doc(GenTenOps GenTenOps Dialects/ -gen-op-doc)
add_mlir_interface(GenTenInterfaces)

set(LLVM_TARGET_DEFINITIONS GenTenOps.td)
mlir_tablegen(GenTenStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(GenTenStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRGenTenStructsIncGen)

