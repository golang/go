// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// Helpers for constructing typed IR nodes.
//
// TODO(mdempsky): Move into their own package so they can be easily
// reused by iimport and frontend optimizations.

type ImplicitNode interface {
	ir.Node
	SetImplicit(x bool)
}

// Implicit returns n after marking it as Implicit.
func Implicit(n ImplicitNode) ImplicitNode {
	n.SetImplicit(true)
	return n
}

// typed returns n after setting its type to typ.
func typed(typ *types.Type, n ir.Node) ir.Node {
	n.SetType(typ)
	n.SetTypecheck(1)
	return n
}

// Values

// FixValue returns val after converting and truncating it as
// appropriate for typ.
func FixValue(typ *types.Type, val constant.Value) constant.Value {
	assert(typ.Kind() != types.TFORW)
	switch {
	case typ.IsInteger():
		val = constant.ToInt(val)
	case typ.IsFloat():
		val = constant.ToFloat(val)
	case typ.IsComplex():
		val = constant.ToComplex(val)
	}
	if !typ.IsUntyped() {
		val = typecheck.ConvertVal(val, typ, false)
	}
	ir.AssertValidTypeForConst(typ, val)
	return val
}

// Expressions

func Addr(pos src.XPos, x ir.Node) *ir.AddrExpr {
	n := typecheck.NodAddrAt(pos, x)
	typed(types.NewPtr(x.Type()), n)
	return n
}

func Deref(pos src.XPos, typ *types.Type, x ir.Node) *ir.StarExpr {
	n := ir.NewStarExpr(pos, x)
	typed(typ, n)
	return n
}

// Statements

func idealType(tv syntax.TypeAndValue) types2.Type {
	// The gc backend expects all expressions to have a concrete type, and
	// types2 mostly satisfies this expectation already. But there are a few
	// cases where the Go spec doesn't require converting to concrete type,
	// and so types2 leaves them untyped. So we need to fix those up here.
	typ := types2.Unalias(tv.Type)
	if basic, ok := typ.(*types2.Basic); ok && basic.Info()&types2.IsUntyped != 0 {
		switch basic.Kind() {
		case types2.UntypedNil:
			// ok; can appear in type switch case clauses
			// TODO(mdempsky): Handle as part of type switches instead?
		case types2.UntypedInt, types2.UntypedFloat, types2.UntypedComplex:
			typ = types2.Typ[types2.Uint]
			if tv.Value != nil {
				s := constant.ToInt(tv.Value)
				assert(s.Kind() == constant.Int)
				if constant.Sign(s) < 0 {
					typ = types2.Typ[types2.Int]
				}
			}
		case types2.UntypedBool:
			typ = types2.Typ[types2.Bool] // expression in "if" or "for" condition
		case types2.UntypedString:
			typ = types2.Typ[types2.String] // argument to "append" or "copy" calls
		case types2.UntypedRune:
			typ = types2.Typ[types2.Int32] // range over rune
		default:
			return nil
		}
	}
	return typ
}

func isTypeParam(t types2.Type) bool {
	_, ok := types2.Unalias(t).(*types2.TypeParam)
	return ok
}

// isNotInHeap reports whether typ is or contains an element of type
// runtime/internal/sys.NotInHeap.
func isNotInHeap(typ types2.Type) bool {
	typ = types2.Unalias(typ)
	if named, ok := typ.(*types2.Named); ok {
		if obj := named.Obj(); obj.Name() == "nih" && obj.Pkg().Path() == "runtime/internal/sys" {
			return true
		}
		typ = named.Underlying()
	}

	switch typ := typ.(type) {
	case *types2.Array:
		return isNotInHeap(typ.Elem())
	case *types2.Struct:
		for i := 0; i < typ.NumFields(); i++ {
			if isNotInHeap(typ.Field(i).Type()) {
				return true
			}
		}
		return false
	default:
		return false
	}
}
