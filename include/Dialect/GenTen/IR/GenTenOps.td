//===-- GenTenOps.td - GEN_TEN dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation set for the GEN_TEN dialect as defined in
// the GEN_TEN specfication (https://some_url/). 
//
//===----------------------------------------------------------------------===//

#ifndef GEN_TEN_OPS
#define GEN_TEN_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"

def GenTen_Dialect : Dialect {
  let name = "gen_ten";
  let cppNamespace = "::mlir::gen_ten";
  let hasConstantMaterializer = 1;
}

class GenTen_Op<string mnemonic, list<Trait> traits = []> :
    Op<GenTen_Dialect, mnemonic, traits>;

def GenTen_Tensor : AnyTypeOf<[I64Tensor, F32Tensor]>;

//===----------------------------------------------------------------------===//
// GenTen Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operator: return
//===----------------------------------------------------------------------===//
def GenTen_ReturnOp : GenTen_Op<"return", [
       Terminator,
       NoSideEffect]> {
  let summary = "return operator";

  let arguments = (ins
    Variadic<GenTen_Tensor>:$inputs
  );
}

//===----------------------------------------------------------------------===//
// Operator: function
//===----------------------------------------------------------------------===//
def GenTen_FunctionOp : GenTen_Op<"function", [  IsolatedFromAbove,
  FunctionOpInterface
  ]> {
  let summary = "function operation";

  let arguments = (ins
    StrAttr:$identifier,
    Variadic<GenTen_Tensor>:$inputs
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  
  let extraClassDeclaration = [{
    static FunctionOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});

    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present.  This can check for preconditions of the
    /// getNumArguments hook not failing.
    // NOTE: No verification
    LogicalResult verifyType() { return success(); }

  }];
}

//===----------------------------------------------------------------------===//
// Operator: constant
//===----------------------------------------------------------------------===//
def GenTen_ConstantOp : GenTen_Op<"constant", [ConstantLike, NoSideEffect,
                                     FirstAttrDerivedResultType]> {
  let summary = "constant op";

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    GenTen_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: identity
//===----------------------------------------------------------------------===//
def GenTen_IdentityOp: GenTen_Op<"identity", [NoSideEffect]> {
  let summary = "identity operator";

  let arguments = (ins
    GenTen_Tensor:$input1
  );

  let results = (outs
    GenTen_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: matmul
//===----------------------------------------------------------------------===//
def GenTen_MatMulOp : GenTen_Op<"matmul", [NoSideEffect]> {
  let summary = "matmul operator";

  let arguments = (ins
    GenTen_Tensor:$a,
    GenTen_Tensor:$b
  );

  let results = (outs
    GenTen_Tensor:$c
  );
}

//===----------------------------------------------------------------------===//
// Operator: transpose
//===----------------------------------------------------------------------===//
def GenTen_TransposeOp : GenTen_Op<"transpose", [NoSideEffect]> {
  let summary = "transpose operator";

  let arguments = (ins
    GenTen_Tensor:$input1,
    I64ArrayAttr:$perms
  );

  let results = (
    outs GenTen_Tensor:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: add
//===----------------------------------------------------------------------===//
def GenTen_AddOp : GenTen_Op<"add", [NoSideEffect, Commutative]> {
  let summary = "add operator";

  let arguments = (ins
    GenTen_Tensor:$input1,
    GenTen_Tensor:$input2
  );

  let results = (outs
    GenTen_Tensor:$output
  );
}

#endif // GEN_TEN_OPS
