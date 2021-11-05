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

func Const(pos src.XPos, typ *types.Type, val constant.Value) ir.Node {
	return typed(typ, ir.NewBasicLit(pos, val))
}

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

func Call(pos src.XPos, typ *types.Type, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots

	if fun.Op() == ir.OTYPE {
		// Actually a type conversion, not a function call.
		if !fun.Type().IsInterface() &&
			(fun.Type().HasTParam() || args[0].Type().HasTParam()) {
			// For type params, we can transform if fun.Type() is known
			// to be an interface (in which case a CONVIFACE node will be
			// inserted). Otherwise, don't typecheck until we actually
			// know the type.
			return typed(typ, n)
		}
		typed(typ, n)
		return transformConvCall(n)
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
		// For most Builtin ops, we delay doing transformBuiltin if any of the
		// args have type params, for a variety of reasons:
		//
		// OMAKE: transformMake can't choose specific ops OMAKESLICE, etc.
		//    until arg type is known
		// OREAL/OIMAG: transformRealImag can't determine type float32/float64
		//    until arg type known
		// OAPPEND: transformAppend requires that the arg is a slice
		// ODELETE: transformDelete requires that the arg is a map
		// OALIGNOF, OSIZEOF: can be eval'ed to a constant until types known.
		switch fun.BuiltinOp {
		case ir.OMAKE, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.ODELETE, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
			hasTParam := false
			for _, arg := range args {
				if fun.BuiltinOp == ir.OOFFSETOF {
					// It's the type of left operand of the
					// selection that matters, not the type of
					// the field itself (which is irrelevant for
					// offsetof).
					arg = arg.(*ir.SelectorExpr).X
				}
				if arg.Type().HasTParam() {
					hasTParam = true
					break
				}
			}
			if hasTParam {
				return typed(typ, n)
			}
		}

		typed(typ, n)
		return transformBuiltin(n)
	}

	// Add information, now that we know that fun is actually being called.
	switch fun := fun.(type) {
	case *ir.SelectorExpr:
		if fun.Op() == ir.OMETHVALUE {
			op := ir.ODOTMETH
			if fun.X.Type().IsInterface() {
				op = ir.ODOTINTER
			}
			fun.SetOp(op)
			// Set the type to include the receiver, since that's what
			// later parts of the compiler expect
			fun.SetType(fun.Selection.Type)
		}
	}

	if fun.Type().HasTParam() || fun.Op() == ir.OXDOT || fun.Op() == ir.OFUNCINST {
		// If the fun arg is or has a type param, we can't do all the
		// transformations, since we may not have needed properties yet
		// (e.g. number of return values, etc). The same applies if a fun
		// which is an XDOT could not be transformed yet because of a generic
		// type in the X of the selector expression.
		//
		// A function instantiation (even if fully concrete) shouldn't be
		// transformed yet, because we need to add the dictionary during the
		// transformation.
		return typed(typ, n)
	}

	// If no type params, do the normal call transformations. This
	// will convert OCALL to OCALLFUNC.
	typed(typ, n)
	transformCall(n)
	return n
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
