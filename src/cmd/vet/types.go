// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gotypes

// This file contains the pieces of the tool that require the go/types package.
// To compile this file, you must first run
//  $ go get code.google.com/p/go.exp/go/types

package main

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.exp/go/types"
)

// Type is equivalent to go/types.Type. Repeating it here allows us to avoid
// depending on the go/types package.
type Type interface {
	String() string
}

func (pkg *Package) check(fs *token.FileSet, astFiles []*ast.File) error {
	pkg.types = make(map[ast.Expr]Type)
	pkg.values = make(map[ast.Expr]interface{})
	exprFn := func(x ast.Expr, typ types.Type, val interface{}) {
		pkg.types[x] = typ
		if val != nil {
			pkg.values[x] = val
		}
	}
	// By providing the Context with our own error function, it will continue
	// past the first error. There is no need for that function to do anything.
	context := types.Context{
		Expr:  exprFn,
		Error: func(error) {},
	}
	_, err := context.Check(fs, astFiles)
	return err
}

// isStruct reports whether the composite literal c is a struct.
// If it is not (probably a struct), it returns a printable form of the type.
func (pkg *Package) isStruct(c *ast.CompositeLit) (bool, string) {
	// Check that the CompositeLit's type is a slice or array (which needs no tag), if possible.
	typ := pkg.types[c]
	// If it's a named type, pull out the underlying type.
	actual := typ
	if namedType, ok := typ.(*types.NamedType); ok {
		actual = namedType.Underlying
	}
	if actual == nil {
		// No type information available. Assume true, so we do the check.
		return true, ""
	}
	switch actual.(type) {
	case *types.Struct:
		return true, typ.String()
	default:
		return false, ""
	}
}

func (f *File) matchArgType(t printfArgType, arg ast.Expr) bool {
	// TODO: for now, we can only test builtin types and untyped constants.
	typ := f.pkg.types[arg]
	if typ == nil {
		return true
	}
	basic, ok := typ.(*types.Basic)
	if !ok {
		return true
	}
	switch basic.Kind {
	case types.Bool:
		return t&argBool != 0
	case types.Int, types.Int8, types.Int16, types.Int32, types.Int64:
		fallthrough
	case types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.Uintptr:
		return t&argInt != 0
	case types.Float32, types.Float64, types.Complex64, types.Complex128:
		return t&argFloat != 0
	case types.String:
		return t&argString != 0
	case types.UnsafePointer:
		return t&(argPointer|argInt) != 0
	case types.UntypedBool:
		return t&argBool != 0
	case types.UntypedComplex:
		return t&argFloat != 0
	case types.UntypedFloat:
		// If it's integral, we can use an int format.
		switch f.pkg.values[arg].(type) {
		case int, int8, int16, int32, int64:
			return t&(argInt|argFloat) != 0
		case uint, uint8, uint16, uint32, uint64:
			return t&(argInt|argFloat) != 0
		}
		return t&argFloat != 0
	case types.UntypedInt:
		return t&argInt != 0
	case types.UntypedRune:
		return t&(argInt|argRune) != 0
	case types.UntypedString:
		return t&argString != 0
	case types.UntypedNil:
		return t&argPointer != 0 // TODO?
	case types.Invalid:
		if *verbose {
			f.Warnf(arg.Pos(), "printf argument %v has invalid or unknown type", arg)
		}
		return true // Probably a type check problem.
	}
	return false
}

// numArgsInSignature tells how many formal arguments the function type
// being called has.
func (f *File) numArgsInSignature(call *ast.CallExpr) int {
	// Check the type of the function or method declaration
	typ := f.pkg.types[call.Fun]
	if typ == nil {
		return 0
	}
	// The type must be a signature, but be sure for safety.
	sig, ok := typ.(*types.Signature)
	if !ok {
		return 0
	}
	return len(sig.Params)
}

// isErrorMethodCall reports whether the call is of a method with signature
//	func Error() string
// where "string" is the universe's string type. We know the method is called "Error".
func (f *File) isErrorMethodCall(call *ast.CallExpr) bool {
	// Is it a selector expression? Otherwise it's a function call, not a method call.
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	// The package is type-checked, so if there are no arguments, we're done.
	if len(call.Args) > 0 {
		return false
	}
	// Check the type of the method declaration
	typ := f.pkg.types[sel]
	if typ == nil {
		return false
	}
	// The type must be a signature, but be sure for safety.
	sig, ok := typ.(*types.Signature)
	if !ok {
		return false
	}
	// There must be a receiver for it to be a method call. Otherwise it is
	// a function, not something that satisfies the error interface.
	if sig.Recv == nil {
		return false
	}
	// There must be no arguments. Already verified by type checking, but be thorough.
	if len(sig.Params) > 0 {
		return false
	}
	// Finally the real questions.
	// There must be one result.
	if len(sig.Results) != 1 {
		return false
	}
	// It must have return type "string" from the universe.
	result := sig.Results[0].Type
	if types.IsIdentical(result, types.Typ[types.String]) {
		return true
	}
	return false
}
