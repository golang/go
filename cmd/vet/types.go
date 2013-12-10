// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the pieces of the tool that use typechecking from the go/types package.

package main

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

func (pkg *Package) check(fs *token.FileSet, astFiles []*ast.File) error {
	pkg.idents = make(map[*ast.Ident]types.Object)
	pkg.spans = make(map[types.Object]Span)
	pkg.types = make(map[ast.Expr]types.Type)
	pkg.values = make(map[ast.Expr]exact.Value)
	// By providing a Config with our own error function, it will continue
	// past the first error. There is no need for that function to do anything.
	config := types.Config{
		Error: func(error) {},
	}
	info := &types.Info{
		Types:   pkg.types,
		Values:  pkg.values,
		Objects: pkg.idents,
	}
	typesPkg, err := config.Check(pkg.path, fs, astFiles, info)
	pkg.typesPkg = typesPkg
	// update spans
	for id, obj := range pkg.idents {
		pkg.growSpan(id, obj)
	}
	return err
}

// isStruct reports whether the composite literal c is a struct.
// If it is not (probably a struct), it returns a printable form of the type.
func (pkg *Package) isStruct(c *ast.CompositeLit) (bool, string) {
	// Check that the CompositeLit's type is a slice or array (which needs no field keys), if possible.
	typ := pkg.types[c]
	// If it's a named type, pull out the underlying type. If it's not, the Underlying
	// method returns the type itself.
	actual := typ
	if actual != nil {
		actual = actual.Underlying()
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

var (
	stringerMethodType = types.New("func() string")
	errorType          = types.New("interface{ Error() string }").(*types.Interface)
	stringerType       = types.New("interface{ String() string }").(*types.Interface)
	// One day this might work. See issue 6259.
	// formatterType   = types.New("interface{Format(f fmt.State, c rune)}")
)

// matchArgType reports an error if printf verb t is not appropriate
// for operand arg.
//
// typ is used only for recursive calls; external callers must supply nil.
//
// (Recursion arises from the compound types {map,chan,slice} which
// may be printed with %d etc. if that is appropriate for their element
// types.)
func (f *File) matchArgType(t printfArgType, typ types.Type, arg ast.Expr) bool {
	return f.matchArgTypeInternal(t, typ, arg, make(map[types.Type]bool))
}

// matchArgTypeInternal is the internal version of matchArgType. It carries a map
// remembering what types are in progress so we don't recur when faced with recursive
// types or mutually recursive types.
func (f *File) matchArgTypeInternal(t printfArgType, typ types.Type, arg ast.Expr, inProgress map[types.Type]bool) bool {
	// %v, %T accept any argument type.
	if t == anyType {
		return true
	}
	if typ == nil {
		// external call
		typ = f.pkg.types[arg]
		if typ == nil {
			return true // probably a type check problem
		}
	}
	// If the type implements fmt.Formatter, we have nothing to check.
	// But (see issue 6259) that's not easy to verify, so instead we see
	// if its method set contains a Format function. We could do better,
	// even now, but we don't need to be 100% accurate. Wait for 6259 to
	// be fixed instead. TODO.
	if hasMethod(typ, "Format") {
		return true
	}
	// If we can use a string, might arg (dynamically) implement the Stringer or Error interface?
	if t&argString != 0 {
		if types.Implements(typ, errorType, false) || types.Implements(typ, stringerType, false) {
			return true
		}
	}

	typ = typ.Underlying()
	if inProgress[typ] {
		// We're already looking at this type. The call that started it will take care of it.
		return true
	}
	inProgress[typ] = true

	switch typ := typ.(type) {
	case *types.Signature:
		return t&argPointer != 0

	case *types.Map:
		// Recur: map[int]int matches %d.
		return t&argPointer != 0 ||
			(f.matchArgTypeInternal(t, typ.Key(), arg, inProgress) && f.matchArgTypeInternal(t, typ.Elem(), arg, inProgress))

	case *types.Chan:
		return t&argPointer != 0

	case *types.Array:
		// Same as slice.
		if types.IsIdentical(typ.Elem().Underlying(), types.Typ[types.Byte]) && t&argString != 0 {
			return true // %s matches []byte
		}
		// Recur: []int matches %d.
		return t&argPointer != 0 || f.matchArgTypeInternal(t, typ.Elem().Underlying(), arg, inProgress)

	case *types.Slice:
		// Same as array.
		if types.IsIdentical(typ.Elem().Underlying(), types.Typ[types.Byte]) && t&argString != 0 {
			return true // %s matches []byte
		}
		// Recur: []int matches %d. But watch out for
		//	type T []T
		// If the element is a pointer type (type T[]*T), it's handled fine by the Pointer case below.
		return t&argPointer != 0 || f.matchArgTypeInternal(t, typ.Elem(), arg, inProgress)

	case *types.Pointer:
		// Ugly, but dealing with an edge case: a known pointer to an invalid type,
		// probably something from a failed import.
		if typ.Elem().String() == "invalid type" {
			if *verbose {
				f.Warnf(arg.Pos(), "printf argument %v is pointer to invalid or unknown type", f.gofmt(arg))
			}
			return true // special case
		}
		// If it's actually a pointer with %p, it prints as one.
		if t == argPointer {
			return true
		}
		// If it's pointer to struct, that's equivalent in our analysis to whether we can print the struct.
		if str, ok := typ.Elem().Underlying().(*types.Struct); ok {
			return f.matchStructArgType(t, str, arg, inProgress)
		}
		// The rest can print with %p as pointers, or as integers with %x etc.
		return t&(argInt|argPointer) != 0

	case *types.Struct:
		return f.matchStructArgType(t, typ, arg, inProgress)

	case *types.Interface:
		// If the static type of the argument is empty interface, there's little we can do.
		// Example:
		//	func f(x interface{}) { fmt.Printf("%s", x) }
		// Whether x is valid for %s depends on the type of the argument to f. One day
		// we will be able to do better. For now, we assume that empty interface is OK
		// but non-empty interfaces, with Stringer and Error handled above, are errors.
		return typ.NumMethods() == 0

	case *types.Basic:
		switch typ.Kind() {
		case types.UntypedBool,
			types.Bool:
			return t&argBool != 0

		case types.UntypedInt,
			types.Int,
			types.Int8,
			types.Int16,
			types.Int32,
			types.Int64,
			types.Uint,
			types.Uint8,
			types.Uint16,
			types.Uint32,
			types.Uint64,
			types.Uintptr:
			return t&argInt != 0

		case types.UntypedFloat,
			types.Float32,
			types.Float64:
			return t&argFloat != 0

		case types.UntypedComplex,
			types.Complex64,
			types.Complex128:
			return t&argComplex != 0

		case types.UntypedString,
			types.String:
			return t&argString != 0

		case types.UnsafePointer:
			return t&(argPointer|argInt) != 0

		case types.UntypedRune:
			return t&(argInt|argRune) != 0

		case types.UntypedNil:
			return t&argPointer != 0 // TODO?

		case types.Invalid:
			if *verbose {
				f.Warnf(arg.Pos(), "printf argument %v has invalid or unknown type", f.gofmt(arg))
			}
			return true // Probably a type check problem.
		}
		panic("unreachable")
	}

	return false
}

// matchStructArgType reports whether all the elements of the struct match the expected
// type. For instance, with "%d" all the elements must be printable with the "%d" format.
func (f *File) matchStructArgType(t printfArgType, typ *types.Struct, arg ast.Expr, inProgress map[types.Type]bool) bool {
	for i := 0; i < typ.NumFields(); i++ {
		if !f.matchArgTypeInternal(t, typ.Field(i).Type(), arg, inProgress) {
			return false
		}
	}
	return true
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
	return sig.Params().Len()
}

// isErrorMethodCall reports whether the call is of a method with signature
//	func Error() string
// where "string" is the universe's string type. We know the method is called "Error".
func (f *File) isErrorMethodCall(call *ast.CallExpr) bool {
	typ := f.pkg.types[call]
	if typ != nil {
		// We know it's called "Error", so just check the function signature.
		return types.IsIdentical(f.pkg.types[call.Fun], stringerMethodType)
	}
	// Without types, we can still check by hand.
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
	typ = f.pkg.types[sel]
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
	if sig.Recv() == nil {
		return false
	}
	// There must be no arguments. Already verified by type checking, but be thorough.
	if sig.Params().Len() > 0 {
		return false
	}
	// Finally the real questions.
	// There must be one result.
	if sig.Results().Len() != 1 {
		return false
	}
	// It must have return type "string" from the universe.
	return sig.Results().At(0).Type() == types.Typ[types.String]
}

// hasMethod reports whether the type contains a method with the given name.
// It is part of the workaround for Formatters and should be deleted when
// that workaround is no longer necessary. TODO: delete when fixed.
func hasMethod(typ types.Type, name string) bool {
	set := typ.MethodSet()
	for i := 0; i < set.Len(); i++ {
		if set.At(i).Obj().Name() == name {
			return true
		}
	}
	return false
}
