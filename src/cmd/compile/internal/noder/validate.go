// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

// match reports whether types t1 and t2 are consistent
// representations for a given expression's type.
func (g *irgen) match(t1 *types.Type, t2 types2.Type, hasOK bool) bool {
	tuple, ok := t2.(*types2.Tuple)
	if !ok {
		// Not a tuple; can use simple type identity comparison.
		return types.Identical(t1, g.typ(t2))
	}

	if hasOK {
		// For has-ok values, types2 represents the expression's type as a
		// 2-element tuple, whereas ir just uses the first type and infers
		// that the second type is boolean. Must match either, since we
		// sometimes delay the transformation to the ir form.
		if tuple.Len() == 2 && types.Identical(t1, g.typ(tuple.At(0).Type())) {
			return true
		}
		return types.Identical(t1, g.typ(t2))
	}

	if t1 == nil || tuple == nil {
		return t1 == nil && tuple == nil
	}
	if !t1.IsFuncArgStruct() {
		return false
	}
	if t1.NumFields() != tuple.Len() {
		return false
	}
	for i, result := range t1.FieldSlice() {
		if !types.Identical(result.Type, g.typ(tuple.At(i).Type())) {
			return false
		}
	}
	return true
}

func (g *irgen) validate(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.CallExpr:
		tv := g.info.Types[n.Fun]
		if tv.IsBuiltin() {
			fun := n.Fun
			for {
				builtin, ok := fun.(*syntax.ParenExpr)
				if !ok {
					break
				}
				fun = builtin.X
			}
			switch builtin := fun.(type) {
			case *syntax.Name:
				g.validateBuiltin(builtin.Value, n)
			case *syntax.SelectorExpr:
				g.validateBuiltin(builtin.Sel.Value, n)
			default:
				g.unhandled("builtin", n)
			}
		}
	}
}

func (g *irgen) validateBuiltin(name string, call *syntax.CallExpr) {
	switch name {
	case "Alignof", "Offsetof", "Sizeof":
		// Check that types2+gcSizes calculates sizes the same
		// as cmd/compile does.

		tv := g.info.Types[call]
		if !tv.IsValue() {
			base.FatalfAt(g.pos(call), "expected a value")
		}

		if tv.Value == nil {
			break // unsafe op is not a constant, so no further validation
		}

		got, ok := constant.Int64Val(tv.Value)
		if !ok {
			base.FatalfAt(g.pos(call), "expected int64 constant value")
		}

		want := g.unsafeExpr(name, call.ArgList[0])
		if got != want {
			base.FatalfAt(g.pos(call), "got %v from types2, but want %v", got, want)
		}
	}
}

// unsafeExpr evaluates the given unsafe builtin function on arg.
func (g *irgen) unsafeExpr(name string, arg syntax.Expr) int64 {
	switch name {
	case "Alignof":
		return g.typ(g.info.Types[arg].Type).Alignment()
	case "Sizeof":
		return g.typ(g.info.Types[arg].Type).Size()
	}

	// Offsetof

	sel := arg.(*syntax.SelectorExpr)
	selection := g.info.Selections[sel]

	typ := g.typ(g.info.Types[sel.X].Type)
	typ = deref(typ)

	var offset int64
	for _, i := range selection.Index() {
		// Ensure field offsets have been calculated.
		types.CalcSize(typ)

		f := typ.Field(i)
		offset += f.Offset
		typ = f.Type
	}
	return offset
}
