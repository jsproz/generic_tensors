//===- GenTenCompilerMlirGen.cpp - MLIR Generation from a GenTen Binary AST -===//
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation from a Module AST
// for the GenTen Binary language.
//
//===----------------------------------------------------------------------===//

#include "Frontend/GenTenCompiler/GenTenCompilerMlirGen.h"

#include <unordered_map>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"

#include "Dialect/GenTen/IR/GenTenOps.h"
#include "Dialect/GenTen/IR/GenTenPrinter.h"
#include "Dialect/GenTen/IR/GenTenCompilerIR.hpp"

namespace {

class GenTenCompilerMlirGenImpl {
  
public:
  GenTenCompilerMlirGenImpl( mlir::MLIRContext &context ) : builder( &context ) { }

  static ::llvm::StringLiteral getOperationName( std::string &operationName ) {
    if ( operationName == "add" )
      return ::llvm::StringLiteral( "gen_ten.add" );
    else if ( operationName == "constant" )
      return ::llvm::StringLiteral( "gen_ten.constant" );
    else if ( operationName == "identity" )
      return ::llvm::StringLiteral( "gen_ten.identity" );
    else if ( operationName == "matmul" )
      return ::llvm::StringLiteral( "gen_ten.matmul" );
    else if ( operationName == "return" )
      return ::llvm::StringLiteral( "gen_ten.return" );
    else if ( operationName == "transpose" )
      return ::llvm::StringLiteral( "gen_ten.transpose" );
    else
      return ::llvm::StringLiteral( "gen_ten.custom" );
  }
  
  llvm::SmallVector<mlir::NamedAttribute, 4> getAttrs( std::shared_ptr<gen_ten_compiler::OperationIR> operationIR ) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    
    for ( auto attributePairIR : operationIR->attributePair ) {
      //      std::cout << attributePairIR->attributeKey << " : " << std::endl;
      
      switch ( attributePairIR->attributeValue->childKind ) {
        case gen_ten_compiler::IR_attributeInt64Value: {
          //          std::cout << "IR_attributeInt64Value" << std::endl;
          mlir::IntegerAttr i64IntegerAttr = builder.getI64IntegerAttr( attributePairIR->attributeValue->getAttributeInt64ValueIR()->int64Value );
          mlir::NamedAttribute namedAttribute = builder.getNamedAttr( attributePairIR->attributeKey, i64IntegerAttr );
          attrs.push_back( namedAttribute );
        }
          break;
        case gen_ten_compiler::IR_attributeInt64ArrayValue: {
          //          mlir::DenseIntElementsAttr denseIntElementsAttr = builder.getI64VectorAttr( attributePairIR->attributeValue->getAttributeInt64ArrayValueIR()->int64Value );
          //          std::cout << "IR_attributeInt64ArrayValue" << std::endl;
          std::vector<mlir::Attribute> attributes;
          for ( int64_t int64Value : attributePairIR->attributeValue->getAttributeInt64ArrayValueIR()->int64Value ) {
            attributes.push_back( builder.getI64IntegerAttr( int64Value ) );
          }
          mlir::ArrayRef<mlir::Attribute> arrayRef( attributes );
          mlir::ArrayAttr arrayAttr = builder.getArrayAttr( arrayRef );
          mlir::NamedAttribute namedAttribute = builder.getNamedAttr( attributePairIR->attributeKey, arrayAttr );
          attrs.push_back( namedAttribute );
        }
          break;
        case gen_ten_compiler::IR_attributeFloatValue: {
          //          std::cout << "IR_attributeFloatValue" << std::endl;
          mlir::FloatAttr f32FloatAttr = builder.getF32FloatAttr( attributePairIR->attributeValue->getAttributeFloatValueIR()->floatValue );
          mlir::NamedAttribute namedAttribute = builder.getNamedAttr( attributePairIR->attributeKey, f32FloatAttr );
          attrs.push_back( namedAttribute );
        }
          break;
        case gen_ten_compiler::IR_attributeFloatArrayValue:
          std::cout << "IR_attributeFloatArrayValue not implemented" << std::endl;
          break;
        case gen_ten_compiler::IR_attributeVariableValue:
          std::cout << "IR_attributeVariableValue not implemented" << std::endl;
          break;
        case gen_ten_compiler::IR_attributeVariableArrayValue:
          std::cout << "IR_attributeVariableArrayValue not implemented" << std::endl;
          break;
        case gen_ten_compiler::IR_attributeFunctionValue:
          std::cout << "IR_attributeFunctionValue not implemented" << std::endl;
          break;
        case gen_ten_compiler::IR_attributeFunctionArrayValue:
          std::cout << "IR_attributeFunctionArrayValue not implemented" << std::endl;
          break;
          
        default:
          break;
      }
    }
    
    return attrs;
  }

  mlir::TensorType getTensorType( std::shared_ptr<gen_ten_compiler::TensorIR> tensorIR ) {
    if ( tensorIR == nullptr ) {
      return mlir::UnrankedTensorType::get( builder.getNoneType() );
    }
    
    if ( tensorIR->dimension.size() > 0 ) {
      llvm::SmallVector<int64_t, 4> shape;
      for ( uint32_t index = 0; index < tensorIR->dimension.size(); ++index ) {
        shape.push_back( tensorIR->dimension[ index ] );
      }
      switch( tensorIR->dataType ) {
        case gen_ten_compiler::DT_unknown:
          return mlir::RankedTensorType::get( shape, builder.getNoneType() );
          break;
        case gen_ten_compiler::DT_float32:
          return mlir::RankedTensorType::get( shape, builder.getF32Type() );
          break;
        default:
          return mlir::RankedTensorType::get( shape, builder.getF32Type() );
          break;
      }
    } else {
      switch( tensorIR->dataType ) {
        case gen_ten_compiler::DT_unknown:
          return mlir::UnrankedTensorType::get( builder.getNoneType() );
          break;
        case gen_ten_compiler::DT_float32:
          return mlir::UnrankedTensorType::get( builder.getF32Type() );
          break;
        case gen_ten_compiler::DT_int64:
          return mlir::UnrankedTensorType::get( builder.getI64Type() );
          break;
        default:
          return mlir::UnrankedTensorType::get( builder.getNoneType() );
          break;
      }
    }
  }
  
  void mlirGen( std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR ) {
    
    std::unordered_map<std::string, mlir::Type> typeLookup;
    
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    for ( auto parameter : functionIR->parameter ) {
      auto tensorType = getTensorType( parameter->tensor );
      inputTypes.push_back( tensorType );
      typeLookup[ parameter->parameterName ] = tensorType;
    }
    llvm::SmallVector<mlir::Type, 4> outputTypes;
    for ( auto result : functionIR->result ) {
      auto tensorType = getTensorType( result->tensor );
      outputTypes.push_back( tensorType );
      typeLookup[ result->resultName ] = tensorType;
    }
    
    auto functionType = builder.getFunctionType( inputTypes, outputTypes );
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    mlir::NamedAttribute attr(  builder.getStringAttr( "identifier" ), builder.getStringAttr( functionIR->functionName ) );
    attrs.push_back(attr);
    auto functionOp = mlir::gen_ten::FunctionOp::create( builder.getUnknownLoc(), functionIR->functionName, functionType, llvm::ArrayRef<mlir::NamedAttribute>( attrs ) );
    module.push_back( functionOp );
    
    mlir::Block *block = functionOp.addEntryBlock();
    builder.setInsertionPointToStart( block );
    
    std::unordered_map<std::string, mlir::Value> valueLookup;
    
    for ( uint32_t index = 0; index < functionIR->parameter.size(); ++index ) {
      valueLookup[ functionIR->parameter[ index ]->parameterName ] = functionOp.getArgument( index );
    }
    
    for ( auto variable : functionIR->variable ) {
      if ( variable->tensor != nullptr && variable->tensor->dataType != gen_ten_compiler::DT_unknown ) {
        auto tensorType = getTensorType( variable->tensor );
        typeLookup[ variable->variableName ] = tensorType;
        
        if ( variable->tensorValue != nullptr ) {
          switch ( variable->tensorValue->childKind ) {
            case gen_ten_compiler::IR_tensorIntValue: {
              std::shared_ptr<gen_ten_compiler::TensorIntValueIR> tensorIntValueIR = variable->tensorValue->getTensorIntValueIR();
              if ( tensorIntValueIR->intValue.size() > 0 ) {
                auto attrib = mlir::DenseIntElementsAttr::get( tensorType, tensorIntValueIR->intValue );
                auto constOp = builder.create<mlir::gen_ten::ConstantOp>( builder.getUnknownLoc(), attrib.getType(), attrib );
                valueLookup[ variable->variableName ] = constOp;
              }
            }
              break;
              
            case gen_ten_compiler::IR_tensorFloatValue: {
              std::shared_ptr<gen_ten_compiler::TensorFloatValueIR> tensorFloatValueIR = variable->tensorValue->getTensorFloatValueIR();
              if ( tensorFloatValueIR->floatValue.size() > 0 ) {
                auto attrib = mlir::DenseFPElementsAttr::get( tensorType, tensorFloatValueIR->floatValue );
                auto constOp = builder.create<mlir::gen_ten::ConstantOp>( builder.getUnknownLoc(), attrib.getType(), attrib );
                valueLookup[ variable->variableName ] = constOp;
              }
            }
              break;
              
            default:
              break;
          }
        }
      }
    }
    
    for ( auto operation : functionIR->operation ) {
      llvm::SmallVector<mlir::Value, 4> inputValues;
      for ( auto inputName : operation->inputName ) {
        if ( valueLookup.find( inputName ) != valueLookup.end() ) {
          inputValues.push_back( valueLookup[ inputName ] );
        } else {
          // TODO MLIR does not support unknown element types?
        }
      }
      llvm::SmallVector<mlir::Type, 4> outputTypes;
      for ( auto outputName : operation->outputName ) {
        if ( typeLookup.find( outputName ) != typeLookup.end() ) {
          outputTypes.push_back( typeLookup[ outputName ] );
        } else {
          // TODO MLIR does not support unknown element types?
        }
      }
      
      llvm::SmallVector<mlir::NamedAttribute, 4> attrs = getAttrs( operation );
      
      mlir::Operation *mlirOperation = nullptr;
      
      llvm::StringRef operationName = getOperationName( operation->operationName );
#if 0
      std::cout << operation->operationName << " -> " << operationName.str() << std::endl;
#endif
      mlir::OperationState state( builder.getUnknownLoc(), operationName );
      state.addTypes( outputTypes );
      state.addOperands( inputValues );
      state.attributes.append( attrs.begin(), attrs.end() );
      //state.addRegion();
      mlirOperation = builder.createOperation( state );
      
      for ( uint32_t index = 0; index < operation->outputName.size(); ++index ) {
        valueLookup[ operation->outputName[ index ] ] = mlirOperation->getResult( index );
      }
#if 0
      mlirOperation->dump();
#endif
    }
    
    llvm::SmallVector<mlir::Type, 4> retTypes;
    llvm::SmallVector<mlir::Value, 4> retValues;
    // change the result of the function operation to the expected values
    unsigned int j = 0;
    for (auto result : functionIR->result) {      
      auto argout = valueLookup[ result->resultName ];
      // NOTE: This is commented out on purpose, the verifier requires empty retTypes for yield
      //retTypes.push_back(argout.getType());
      retValues.push_back(argout);
      ++j;
    }
    assert(j == functionOp.getNumResults() && "Number of results must match" );
    
    builder.create<::mlir::gen_ten::ReturnOp>(builder.getUnknownLoc(), retTypes, retValues);
  }
  
  mlir::OwningOpRef<mlir::ModuleOp> mlirGen( std::shared_ptr<gen_ten_compiler::ProgramIR> programIR ) {
    module = mlir::ModuleOp::create( builder.getUnknownLoc() );
    
    std::shared_ptr<gen_ten_compiler::FunctionIR> functionIR = programIR->function;
    mlirGen( functionIR );
    
    return module;
  }
  
private:
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  
};

} // namespace

namespace gen_ten_compiler {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen( mlir::MLIRContext &context, std::shared_ptr<gen_ten_compiler::ProgramIR> programIR ) {
  GenTenCompilerMlirGenImpl genTenCompilerMlirGenImpl( context );
  mlir::OwningOpRef<mlir::ModuleOp> module = genTenCompilerMlirGenImpl.mlirGen( programIR );
  
  return module;
}

} // namespace gen_ten_compiler
