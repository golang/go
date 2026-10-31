// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"go/token"
	"reflect"
)

// Equal reports whether two nodes are structurally equal,
// ignoring fields of type [token.Pos], [ast.Object],
// and [ast.Scope], and comments.
//
// The operands x and y may be nil.
// A nil slice is not equal to an empty slice.
//
// The provided function determines whether two identifiers
// should be considered identical.
func Equal(x, y ast.Node, identical func(x, y *ast.Ident) bool) bool {
	if x == nil || y == nil {
		return x == y
	}
	return equal(reflect.ValueOf(x), reflect.ValueOf(y), identical)
}

// EqualSyntax reports whether x and y are equal.
// Identifiers are considered equal if they are spelled the same.
// Comments are ignored.
func EqualSyntax(x, y ast.Expr) bool {
	sameName := func(x, y *ast.Ident) bool { return x.Name == y.Name }
	return Equal(x, y, sameName)
}

func equal(x, y reflect.Value, identical func(x, y *ast.Ident) bool) bool {
	// Ensure types are the same
	if x.Type() != y.Type() {
		return false
	}
	switch x.Kind() {
	case reflect.Pointer:
		if x.IsNil() || y.IsNil() {
			return x.IsNil() == y.IsNil()
		}
		switch t := x.Interface().(type) {
		// Skip fields of types potentially involved in cycles.
		case *ast.Object, *ast.Scope, *ast.CommentGroup:
			return true
		case *ast.Ident:
			return identical(t, y.Interface().(*ast.Ident))
		default:
			return equal(x.Elem(), y.Elem(), identical)
		}

	case reflect.Interface:
		if x.IsNil() || y.IsNil() {
			return x.IsNil() == y.IsNil()
		}
		return equal(x.Elem(), y.Elem(), identical)

	case reflect.Struct:
		for i := range x.NumField() {
			xf := x.Field(i)
			yf := y.Field(i)
			// Skip position fields.
			if xpos, ok := xf.Interface().(token.Pos); ok {
				ypos := yf.Interface().(token.Pos)
				// Numeric value of a Pos is not significant but its "zeroness" is,
				// because it is often significant, e.g. CallExpr.Variadic(Ellipsis), ChanType.Arrow.
				if xpos.IsValid() != ypos.IsValid() {
					return false
				}
			} else if !equal(xf, yf, identical) {
				return false
			}
		}
		return true

	case reflect.Slice:
		if x.IsNil() || y.IsNil() {
			return x.IsNil() == y.IsNil()
		}
		if x.Len() != y.Len() {
			return false
		}
		for i := range x.Len() {
			if !equal(x.Index(i), y.Index(i), identical) {
				return false
			}
		}
		return true

	case reflect.String:
		return x.String() == y.String()

	case reflect.Bool:
		return x.Bool() == y.Bool()

	case reflect.Int:
		return x.Int() == y.Int()

	default:
		panic(x)
	}
}
