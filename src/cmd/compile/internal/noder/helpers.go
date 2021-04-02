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

func Nil(pos src.XPos, typ *types.Type) ir.Node {
	return typed(typ, ir.NewNilExpr(pos))
}

// Expressions

func Addr(pos src.XPos, x ir.Node) *ir.AddrExpr {
	n := typecheck.NodAddrAt(pos, x)
	switch x.Op() {
	case ir.OARRAYLIT, ir.OMAPLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
		n.SetOp(ir.OPTRLIT)
	}
	typed(types.NewPtr(x.Type()), n)
	return n
}

func Assert(pos src.XPos, x ir.Node, typ *types.Type) ir.Node {
	return typed(typ, ir.NewTypeAssertExpr(pos, x, nil))
}

func Binary(pos src.XPos, op ir.Op, typ *types.Type, x, y ir.Node) ir.Node {
	switch op {
	case ir.OANDAND, ir.OOROR:
		return typed(x.Type(), ir.NewLogicalExpr(pos, op, x, y))
	case ir.OADD:
		n := ir.NewBinaryExpr(pos, op, x, y)
		if x.Type().HasTParam() || y.Type().HasTParam() {
			// Delay transformAdd() if either arg has a type param,
			// since it needs to know the exact types to decide whether
			// to transform OADD to OADDSTR.
			n.SetType(typ)
			n.SetTypecheck(3)
			return n
		}
		typed(typ, n)
		return transformAdd(n)
	default:
		return typed(x.Type(), ir.NewBinaryExpr(pos, op, x, y))
	}
}

func Call(pos src.XPos, typ *types.Type, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots
	// n.Use will be changed to ir.CallUseStmt in g.stmt() if this call is
	// just a statement (any return values are ignored).
	n.Use = ir.CallUseExpr

	if fun.Op() == ir.OTYPE {
		// Actually a type conversion, not a function call.
		if fun.Type().HasTParam() || args[0].Type().HasTParam() {
			// For type params, don't typecheck until we actually know
			// the type.
			return typed(typ, n)
		}
		typed(typ, n)
		return transformConvCall(n)
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
		// For Builtin ops, we currently stay with using the old
		// typechecker to transform the call to a more specific expression
		// and possibly use more specific ops. However, for a bunch of the
		// ops, we delay doing the old typechecker if any of the args have
		// type params, for a variety of reasons:
		//
		// OMAKE: hard to choose specific ops OMAKESLICE, etc. until arg type is known
		// OREAL/OIMAG: can't determine type float32/float64 until arg type know
		// OLEN/OCAP: old typechecker will complain if arg is not obviously a slice/array.
		// OAPPEND: old typechecker will complain if arg is not obviously slice, etc.
		//
		// We will eventually break out the transforming functionality
		// needed for builtin's, and call it here or during stenciling, as
		// appropriate.
		switch fun.BuiltinOp {
		case ir.OMAKE, ir.OREAL, ir.OIMAG, ir.OLEN, ir.OCAP, ir.OAPPEND:
			hasTParam := false
			for _, arg := range args {
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
	case *ir.ClosureExpr:
		fun.Func.SetClosureCalled(true)
	case *ir.SelectorExpr:
		if fun.Op() == ir.OCALLPART {
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

	if fun.Type().HasTParam() {
		// If the fun arg is or has a type param, don't do any extra
		// transformations, since we may not have needed properties yet
		// (e.g. number of return values, etc). The type param is probably
		// described by a structural constraint that requires it to be a
		// certain function type, etc., but we don't want to analyze that.
		return typed(typ, n)
	}

	if fun.Op() == ir.OXDOT {
		if !fun.(*ir.SelectorExpr).X.Type().HasTParam() {
			base.FatalfAt(pos, "Expecting type param receiver in %v", fun)
		}
		// For methods called in a generic function, don't do any extra
		// transformations. We will do those later when we create the
		// instantiated function and have the correct receiver type.
		typed(typ, n)
		return n
	}
	if fun.Op() != ir.OFUNCINST {
		// If no type params, do the normal call transformations. This
		// will convert OCALL to OCALLFUNC.
		typed(typ, n)
		transformCall(n)
		return n
	}

	// Leave the op as OCALL, which indicates the call still needs typechecking.
	typed(typ, n)
	return n
}

func Compare(pos src.XPos, typ *types.Type, op ir.Op, x, y ir.Node) ir.Node {
	n := ir.NewBinaryExpr(pos, op, x, y)
	if x.Type().HasTParam() || y.Type().HasTParam() {
		// Delay transformCompare() if either arg has a type param, since
		// it needs to know the exact types to decide on any needed conversions.
		n.SetType(typ)
		n.SetTypecheck(3)
		return n
	}
	typed(typ, n)
	transformCompare(n)
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
	return dot(pos, typ, ir.OCALLPART, x, method)
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

func Index(pos src.XPos, typ *types.Type, x, index ir.Node) ir.Node {
	n := ir.NewIndexExpr(pos, x, index)
	if x.Type().HasTParam() {
		// transformIndex needs to know exact type
		n.SetType(typ)
		n.SetTypecheck(3)
		return n
	}
	typed(typ, n)
	// transformIndex will modify n.Type() for OINDEXMAP.
	transformIndex(n)
	return n
}

func Slice(pos src.XPos, typ *types.Type, x, low, high, max ir.Node) ir.Node {
	op := ir.OSLICE
	if max != nil {
		op = ir.OSLICE3
	}
	n := ir.NewSliceExpr(pos, op, x, low, high, max)
	if x.Type().HasTParam() {
		// transformSlice needs to know if x.Type() is a string or an array or a slice.
		n.SetType(typ)
		n.SetTypecheck(3)
		return n
	}
	typed(typ, n)
	transformSlice(n)
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
	return ir.NewAssignOpStmt(pos, op, x, typecheck.DefaultLit(ir.NewBasicLit(pos, one), x.Type()))
}
