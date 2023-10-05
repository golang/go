// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file defines various common helpers.

import (
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"strings"

	"golang.org/x/tools/internal/typeparams"
)

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}

// TODO(adonovan): use go1.21's slices.Clone.
func clone[T any](slice []T) []T { return append([]T{}, slice...) }

// TODO(adonovan): use go1.21's slices.Index.
func index[T comparable](slice []T, x T) int {
	for i, elem := range slice {
		if elem == x {
			return i
		}
	}
	return -1
}

func btoi(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

func offsetOf(fset *token.FileSet, pos token.Pos) int {
	return fset.PositionFor(pos, false).Offset
}

// objectKind returns an object's kind (e.g. var, func, const, typename).
func objectKind(obj types.Object) string {
	return strings.TrimPrefix(strings.ToLower(reflect.TypeOf(obj).String()), "*types.")
}

// within reports whether pos is within the half-open interval [n.Pos, n.End).
func within(pos token.Pos, n ast.Node) bool {
	return n.Pos() <= pos && pos < n.End()
}

// trivialConversion reports whether it is safe to omit the implicit
// value-to-variable conversion that occurs in argument passing or
// result return. The only case currently allowed is converting from
// untyped constant to its default type (e.g. 0 to int).
//
// The reason for this check is that converting from A to B to C may
// yield a different result than converting A directly to C: consider
// 0 to int32 to any.
func trivialConversion(val types.Type, obj *types.Var) bool {
	return types.Identical(types.Default(val), obj.Type())
}

func checkInfoFields(info *types.Info) {
	assert(info.Defs != nil, "types.Info.Defs is nil")
	assert(info.Implicits != nil, "types.Info.Implicits is nil")
	assert(info.Scopes != nil, "types.Info.Scopes is nil")
	assert(info.Selections != nil, "types.Info.Selections is nil")
	assert(info.Types != nil, "types.Info.Types is nil")
	assert(info.Uses != nil, "types.Info.Uses is nil")
}

func funcHasTypeParams(decl *ast.FuncDecl) bool {
	// generic function?
	if decl.Type.TypeParams != nil {
		return true
	}
	// method on generic type?
	if decl.Recv != nil {
		t := decl.Recv.List[0].Type
		if u, ok := t.(*ast.StarExpr); ok {
			t = u.X
		}
		return is[*ast.IndexExpr](t) || is[*ast.IndexListExpr](t)
	}
	return false
}

// intersects reports whether the maps' key sets intersect.
func intersects[K comparable, T1, T2 any](x map[K]T1, y map[K]T2) bool {
	if len(x) > len(y) {
		return intersects(y, x)
	}
	for k := range x {
		if _, ok := y[k]; ok {
			return true
		}
	}
	return false
}

// convert returns syntax for the conversion T(x).
func convert(T, x ast.Expr) *ast.CallExpr {
	// The formatter generally adds parens as needed,
	// but before go1.22 it had a bug (#63362) for
	// channel types that requires this workaround.
	if ch, ok := T.(*ast.ChanType); ok && ch.Dir == ast.RECV {
		T = &ast.ParenExpr{X: T}
	}
	return &ast.CallExpr{
		Fun:  T,
		Args: []ast.Expr{x},
	}
}

// isPointer reports whether t is a pointer type.
func isPointer(t types.Type) bool { return t != deref(t) }

// indirectSelection is like seln.Indirect() without bug #8353.
func indirectSelection(seln *types.Selection) bool {
	// Work around bug #8353 in Selection.Indirect when Kind=MethodVal.
	if seln.Kind() == types.MethodVal {
		tArg, indirect := effectiveReceiver(seln)
		if indirect {
			return true
		}

		tParam := seln.Obj().Type().(*types.Signature).Recv().Type()
		return isPointer(tArg) && !isPointer(tParam) // implicit *
	}

	return seln.Indirect()
}

// effectiveReceiver returns the effective type of the method
// receiver after all implicit field selections (but not implicit * or
// & operations) have been applied.
//
// The boolean indicates whether any implicit field selection was indirect.
func effectiveReceiver(seln *types.Selection) (types.Type, bool) {
	assert(seln.Kind() == types.MethodVal, "not MethodVal")
	t := seln.Recv()
	indices := seln.Index()
	indirect := false
	for _, index := range indices[:len(indices)-1] {
		if tElem := deref(t); tElem != t {
			indirect = true
			t = tElem
		}
		t = typeparams.CoreType(t).(*types.Struct).Field(index).Type()
	}
	return t, indirect
}
