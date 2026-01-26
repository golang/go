// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file defines various common helpers.

import (
	"go/ast"
	"go/constant"
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
//
// trivialConversion under-approximates trivial conversions, as unfortunately
// go/types does not record the type of an expression *before* it is implicitly
// converted, and therefore it cannot distinguish typed constant
// expressions from untyped constant expressions. For example, in the
// expression `c + 2`, where c is a uint32 constant, trivialConversion does not
// detect that the default type of this expression is actually uint32, not untyped
// int.
//
// We could, of course, do better here by reverse engineering some of go/types'
// constant handling. That may or may not be worthwhile.
//
// Example: in func f() int32 { return 0 },
// the type recorded for 0 is int32, not untyped int;
// although it is Identical to the result var,
// the conversion is non-trivial.
func trivialConversion(fromValue constant.Value, from, to types.Type) bool {
	if fromValue != nil {
		var defaultType types.Type
		switch fromValue.Kind() {
		case constant.Bool:
			defaultType = types.Typ[types.Bool]
		case constant.String:
			defaultType = types.Typ[types.String]
		case constant.Int:
			defaultType = types.Typ[types.Int]
		case constant.Float:
			defaultType = types.Typ[types.Float64]
		case constant.Complex:
			defaultType = types.Typ[types.Complex128]
		default:
			return false
		}
		return types.Identical(defaultType, to)
	}
	return types.Identical(from, to)
}

func checkInfoFields(info *types.Info) {
	assert(info.Defs != nil, "types.Info.Defs is nil")
	assert(info.Implicits != nil, "types.Info.Implicits is nil")
	assert(info.Scopes != nil, "types.Info.Scopes is nil")
	assert(info.Selections != nil, "types.Info.Selections is nil")
	assert(info.Types != nil, "types.Info.Types is nil")
	assert(info.Uses != nil, "types.Info.Uses is nil")
	assert(info.FileVersions != nil, "types.Info.FileVersions is nil")
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

// isPointer reports whether t's core type is a pointer.
func isPointer(t types.Type) bool {
	return is[*types.Pointer](typeparams.CoreType(t))
}

// indirectSelection is like seln.Indirect() without bug #8353.
func indirectSelection(seln *types.Selection) bool {
	// Work around bug #8353 in Selection.Indirect when Kind=MethodVal.
	if seln.Kind() == types.MethodVal {
		tArg, indirect := effectiveReceiver(seln)
		if indirect {
			return true
		}

		tParam := seln.Obj().Type().Underlying().(*types.Signature).Recv().Type()
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
		if isPointer(t) {
			indirect = true
			t = typeparams.MustDeref(t)
		}
		t = typeparams.CoreType(t).(*types.Struct).Field(index).Type()
	}
	return t, indirect
}
