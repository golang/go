// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
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

func OrigConst(pos src.XPos, typ *types.Type, val constant.Value, op ir.Op, raw string) ir.Node {
	orig := ir.NewRawOrigExpr(pos, op, raw)
	return ir.NewConstExpr(val, typed(typ, orig))
}

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
		val = typecheck.DefaultLit(ir.NewBasicLit(src.NoXPos, val), typ).Val()
	}
	if !typ.IsTypeParam() {
		ir.AssertValidTypeForConst(typ, val)
	}
	return val
}

func Nil(pos src.XPos, typ *types.Type) ir.Node {
	return typed(typ, ir.NewNilExpr(pos))
}

// Expressions

func Addr(pos src.XPos, x ir.Node) *ir.AddrExpr {
	n := typecheck.NodAddrAt(pos, x)
	typed(types.NewPtr(x.Type()), n)
	return n
}

func Assert(pos src.XPos, x ir.Node, typ *types.Type) ir.Node {
	return typed(typ, ir.NewTypeAssertExpr(pos, x, nil))
}

func Binary(pos src.XPos, op ir.Op, typ *types.Type, x, y ir.Node) *ir.BinaryExpr {
	switch op {
	case ir.OADD:
		n := ir.NewBinaryExpr(pos, op, x, y)
		typed(typ, n)
		return n
	default:
		n := ir.NewBinaryExpr(pos, op, x, y)
		typed(x.Type(), n)
		return n
	}
}

func Compare(pos src.XPos, typ *types.Type, op ir.Op, x, y ir.Node) *ir.BinaryExpr {
	n := ir.NewBinaryExpr(pos, op, x, y)
	typed(typ, n)
	return n
}

func Deref(pos src.XPos, typ *types.Type, x ir.Node) *ir.StarExpr {
	n := ir.NewStarExpr(pos, x)
	typed(typ, n)
	return n
}

func DotField(pos src.XPos, x ir.Node, index int) *ir.SelectorExpr {
	op, typ := ir.ODOT, x.Type()
	if typ.IsPtr() {
		op, typ = ir.ODOTPTR, typ.Elem()
	}
	if !typ.IsStruct() {
		base.FatalfAt(pos, "DotField of non-struct: %L", x)
	}

	// TODO(mdempsky): This is the backend's responsibility.
	types.CalcSize(typ)

	field := typ.Field(index)
	return dot(pos, field.Type, op, x, field)
}

func DotMethod(pos src.XPos, x ir.Node, index int) *ir.SelectorExpr {
	method := method(x.Type(), index)

	// Method value.
	typ := typecheck.NewMethodType(method.Type, nil)
	return dot(pos, typ, ir.OMETHVALUE, x, method)
}

// MethodExpr returns a OMETHEXPR node with the indicated index into the methods
// of typ. The receiver type is set from recv, which is different from typ if the
// method was accessed via embedded fields. Similarly, the X value of the
// ir.SelectorExpr is recv, the original OTYPE node before passing through the
// embedded fields.
func MethodExpr(pos src.XPos, recv ir.Node, embed *types.Type, index int) *ir.SelectorExpr {
	method := method(embed, index)
	typ := typecheck.NewMethodType(method.Type, recv.Type())
	// The method expression T.m requires a wrapper when T
	// is different from m's declared receiver type. We
	// normally generate these wrappers while writing out
	// runtime type descriptors, which is always done for
	// types declared at package scope. However, we need
	// to make sure to generate wrappers for anonymous
	// receiver types too.
	if recv.Sym() == nil {
		typecheck.NeedRuntimeType(recv.Type())
	}
	return dot(pos, typ, ir.OMETHEXPR, recv, method)
}

func dot(pos src.XPos, typ *types.Type, op ir.Op, x ir.Node, selection *types.Field) *ir.SelectorExpr {
	n := ir.NewSelectorExpr(pos, op, x, selection.Sym)
	n.Selection = selection
	typed(typ, n)
	return n
}

// TODO(mdempsky): Move to package types.
func method(typ *types.Type, index int) *types.Field {
	if typ.IsInterface() {
		return typ.AllMethods().Index(index)
	}
	return types.ReceiverBaseType(typ).Methods().Index(index)
}

func Index(pos src.XPos, typ *types.Type, x, index ir.Node) *ir.IndexExpr {
	n := ir.NewIndexExpr(pos, x, index)
	typed(typ, n)
	return n
}

func Slice(pos src.XPos, typ *types.Type, x, low, high, max ir.Node) *ir.SliceExpr {
	op := ir.OSLICE
	if max != nil {
		op = ir.OSLICE3
	}
	n := ir.NewSliceExpr(pos, op, x, low, high, max)
	typed(typ, n)
	return n
}

func Unary(pos src.XPos, typ *types.Type, op ir.Op, x ir.Node) ir.Node {
	switch op {
	case ir.OADDR:
		return Addr(pos, x)
	case ir.ODEREF:
		return Deref(pos, typ, x)
	}

	if op == ir.ORECV {
		if typ.IsFuncArgStruct() && typ.NumFields() == 2 {
			// Remove the second boolean type (if provided by type2),
			// since that works better with the rest of the compiler
			// (which will add it back in later).
			assert(typ.Field(1).Type.Kind() == types.TBOOL)
			typ = typ.Field(0).Type
		}
	}
	return typed(typ, ir.NewUnaryExpr(pos, op, x))
}

// Statements

var one = constant.MakeInt64(1)

func IncDec(pos src.XPos, op ir.Op, x ir.Node) *ir.AssignOpStmt {
	assert(x.Type() != nil)
	bl := ir.NewBasicLit(pos, one)
	if x.Type().HasTParam() {
		// If the operand is generic, then types2 will have proved it must be
		// a type that fits with increment/decrement, so just set the type of
		// "one" to n.Type(). This works even for types that are eventually
		// float or complex.
		typed(x.Type(), bl)
	} else {
		bl = typecheck.DefaultLit(bl, x.Type())
	}
	return ir.NewAssignOpStmt(pos, op, x, bl)
}

func idealType(tv types2.TypeAndValue) types2.Type {
	// The gc backend expects all expressions to have a concrete type, and
	// types2 mostly satisfies this expectation already. But there are a few
	// cases where the Go spec doesn't require converting to concrete type,
	// and so types2 leaves them untyped. So we need to fix those up here.
	typ := tv.Type
	if basic, ok := typ.(*types2.Basic); ok && basic.Info()&types2.IsUntyped != 0 {
		switch basic.Kind() {
		case types2.UntypedNil:
			// ok; can appear in type switch case clauses
			// TODO(mdempsky): Handle as part of type switches instead?
		case types2.UntypedInt, types2.UntypedFloat, types2.UntypedComplex:
			// Untyped rhs of non-constant shift, e.g. x << 1.0.
			// If we have a constant value, it must be an int >= 0.
			if tv.Value != nil {
				s := constant.ToInt(tv.Value)
				assert(s.Kind() == constant.Int && constant.Sign(s) >= 0)
			}
			typ = types2.Typ[types2.Uint]
		case types2.UntypedBool:
			typ = types2.Typ[types2.Bool] // expression in "if" or "for" condition
		case types2.UntypedString:
			typ = types2.Typ[types2.String] // argument to "append" or "copy" calls
		default:
			return nil
		}
	}
	return typ
}

func isTypeParam(t types2.Type) bool {
	_, ok := t.(*types2.TypeParam)
	return ok
}

// isNotInHeap reports whether typ is or contains an element of type
// runtime/internal/sys.NotInHeap.
func isNotInHeap(typ types2.Type) bool {
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
