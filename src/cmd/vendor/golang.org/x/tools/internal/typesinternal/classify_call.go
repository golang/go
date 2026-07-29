// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/ast"
	"go/types"
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
	tv := info.Types[call.Fun]
	if tv.IsType() {
		return CallConversion
	}
	if tv.IsBuiltin() {
		return CallBuiltin
	}
	id := UsedIdent(info, call.Fun)
	if id == nil {
		return CallDynamic
	}
	obj := info.Uses[id]
	// Classify the call by the type of the object, if any.
	switch obj := obj.(type) {
	case *types.Func:
		if isInterfaceMethod(obj) {
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
	if info.Types == nil || info.Uses == nil {
		panic("one of info.Types or info.Uses is nil; both must be populated")
	}
	// Look through type instantiation if necessary.
	switch d := ast.Unparen(e).(type) {
	case *ast.IndexExpr:
		if info.Types[d.Index].IsType() {
			e = d.X
		}
	case *ast.IndexListExpr:
		e = d.X
	}

	switch e := ast.Unparen(e).(type) {
	// info.Uses always has the object we want, even for selector expressions.
	// We don't need info.Selections.
	// See go/types/recording.go:recordSelection.
	case *ast.Ident:
		return e
	case *ast.SelectorExpr:
		return e.Sel
	}
	return nil
}

// See [golang.org/x/tools/go/types/typeutil.Callee].
func Callee(info *types.Info, call *ast.CallExpr) types.Object {
	id := UsedIdent(info, call.Fun)
	if id == nil {
		return nil
	}
	obj := info.Uses[id]
	if obj == nil {
		return nil
	}
	if _, ok := obj.(*types.TypeName); ok {
		return nil
	}
	return obj
}

// See [golang.org/x/tools/go/types/typeutil.StaticCallee].
func StaticCallee(info *types.Info, call *ast.CallExpr) *types.Func {
	id := UsedIdent(info, call.Fun)
	if id == nil {
		return nil
	}
	obj := info.Uses[id]
	if obj == nil {
		return nil
	}
	fn, _ := obj.(*types.Func)
	if fn == nil || isInterfaceMethod(fn) {
		return nil
	}
	return fn
}

// isInterfaceMethod reports whether its argument is a method of an interface.
func isInterfaceMethod(f *types.Func) bool {
	recv := f.Signature().Recv()
	return recv != nil && types.IsInterface(recv.Type())
}
