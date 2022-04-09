# Generic Tensors

This is an out-of-tree MLIR frontend for a simple text based language for tensors. It is just a
demonstration and not particularly useful. It does very little error checking.

## Docker

### Running Ubuntu

```bash
$ docker run \
--name ubuntu \
-e HOST_IP=$(ifconfig en0 | awk '/ *inet /{print $2}') \
-v /Users/<user-name>/gen_ten:/gen_ten \
-t -i \
ubuntu /bin/bash
```

### Updating apt-get

```bash
$ apt-get update
```

### Installing Tools

```bash
$ apt-get install -y build-essential git cmake clang llvm
```

## Getting and Building MLIR

Checkout hash is last tested, newer ones may work too.

```bash
cd /gen-ten
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout ff758372bd51840b4f566968fb0929d19557dd9b
mkdir build
cd build
cmake -G "Unix Makefiles" ../llvm \
-DLLVM_ENABLE_PROJECTS="clang;mlir;libcxx;libcxxabi" \
-DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_RTTI=ON \
-DCMAKE_C_COMPILER=/usr/bin/clang \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++
cmake --build . -j8
cmake --build . --target check-mlir
export LLVM_PROJ_BUILD=$PWD
```

## Building generic-tensors

This setup assumes that you have built LLVM and MLIR in `$LLVM_PROJ_BUILD`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G "Unix Makefiles" .. -DMLIR_DIR=$LLVM_PROJ_BUILD/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_PROJ_BUILD/bin/llvm-lit
cmake --build . -j8
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

## Using generic-tensors

```
Usage
    generic-tensors -dump-tokens <input_file>
    generic-tensors -dump-parsed <input_file>
    generic-tensors -dump-mlir [-clean-up] <input_file>
    generic-tensors -help
```

Use `-dump-tokens` to generate a list of tokens scanned from the input file.
Use `-dump-parsed` to pretty print the parsed input file.
Use `-dump-mlir` to get a dump of the MLIR.

## Example

There is an example file in the examples folder.

```swift
%func Gemm( a: Float32[2*3], b: Float32[3*2], c: Float32[2*2] )
      -> ( y: Float32[2*2] )
{
  
  ( transposeA ) = transpose( a ) [ "perms" : [ 1, 0 ] ]
  ( transposeB ) = transpose( b ) [ "perms" : [ 1, 0 ] ]
  ( matMulTransposeATransposeB ) = matmul( transposeA, transposeB )
  ( matMulMatMulTransposeATransposeBAlpha ) = matmul( matMulTransposeATransposeB, alpha )
  ( matMulCBeta ) = matmul( c, beta )
  ( result ) = add( matMulMatMulTransposeATransposeBAlpha, matMulCBeta )
  ( y ) = identity( result )
  
  %var transposeA: Float32[3*2]
  %var transposeB: Float32[2*3]
  %var alpha: Float32[1] = [
    1
  ]
  %var beta: Float32[1] = [
    1
  ]
  %var matMulTransposeATransposeB: Float32[2*2]
  %var matMulMatMulTransposeATransposeBAlpha: Float32[2*2]
  %var matMulCBeta: Float32[2*2]
  %var result: Float32[2*2]
}
```
