// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/ast"
	"go/types"
	_ "unsafe"
)

// CallKind describes the function position of an [*ast.CallExpr].
type CallKind int

const (
	CallStatic     CallKind = iota // static call to known function
	CallInterface                  // dynamic call through an interface method
	CallDynamic                    // dynamic call of a func value
	CallBuiltin                    // call to a builtin function
	CallConversion                 // a conversion (not a call)
)

var callKindNames = []string{
	"CallStatic",
	"CallInterface",
	"CallDynamic",
	"CallBuiltin",
	"CallConversion",
}

func (k CallKind) String() string {
	if i := int(k); i >= 0 && i < len(callKindNames) {
		return callKindNames[i]
	}
	return fmt.Sprintf("typeutil.CallKind(%d)", k)
}

// ClassifyCall classifies the function position of a call expression ([*ast.CallExpr]).
// It distinguishes among true function calls, calls to builtins, and type conversions,
// and further classifies function calls as static calls (where the function is known),
// dynamic interface calls, and other dynamic calls.
//
// For the declarations:
//
//	func f() {}
//	func g[T any]() {}
//	var v func()
//	var s []func()
//	type I interface { M() }
//	var i I
//
// ClassifyCall returns the following:
//
//	f()           CallStatic
//	g[int]()      CallStatic
//	i.M()         CallInterface
//	min(1, 2)     CallBuiltin
//	v()           CallDynamic
//	s[0]()        CallDynamic
//	int(x)        CallConversion
//	[]byte("")    CallConversion
func ClassifyCall(info *types.Info, call *ast.CallExpr) CallKind {
	if info.Types == nil {
		panic("ClassifyCall: info.Types is nil")
	}
	if info.Types[call.Fun].IsType() {
		return CallConversion
	}
	obj := info.Uses[UsedIdent(info, call.Fun)]
	// Classify the call by the type of the object, if any.
	switch obj := obj.(type) {
	case *types.Builtin:
		return CallBuiltin
	case *types.Func:
		if interfaceMethod(obj) {
			return CallInterface
		}
		return CallStatic
	default:
		return CallDynamic
	}
}

// UsedIdent returns the identifier such that info.Uses[UsedIdent(info, e)]
// is the [types.Object] used by e, if any.
//
// If e is one of various forms of reference:
//
//	f, c, v, T           lexical reference
//	pkg.X                qualified identifier
//	f[T] or pkg.F[K,V]   instantiations of the above kinds
//	expr.f               field or method value selector
//	T.f                  method expression selector
//
// UsedIdent returns the identifier whose is associated value in [types.Info.Uses]
// is the object to which it refers.
//
// For the declarations:
//
//	func F[T any] {...}
//	type I interface { M() }
//	var (
//	  x int
//	  s struct { f  int }
//	  a []int
//	  i I
//	)
//
// UsedIdent returns the following:
//
//	Expr          UsedIdent
//	x             x
//	s.f           f
//	F[int]        F
//	i.M           M
//	I.M           M
//	min           min
//	int           int
//	1             nil
//	a[0]          nil
//	[]byte        nil
//
// Note: if e is an instantiated function or method, UsedIdent returns
// the corresponding generic function or method on the generic type.
func UsedIdent(info *types.Info, e ast.Expr) *ast.Ident {
	return usedIdent(info, e)
}

//go:linkname usedIdent golang.org/x/tools/go/types/typeutil.usedIdent
func usedIdent(info *types.Info, e ast.Expr) *ast.Ident

//go:linkname interfaceMethod golang.org/x/tools/go/types/typeutil.interfaceMethod
func interfaceMethod(f *types.Func) bool
