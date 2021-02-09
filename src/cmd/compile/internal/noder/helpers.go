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
//
// TODO(mdempsky): Update to consistently return already typechecked
// results, rather than leaving the caller responsible for using
// typecheck.Expr or typecheck.Stmt.

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
	// TODO(mdempsky): Avoid typecheck.Expr. Probably just need to set OPTRLIT when appropriate.
	n := typecheck.Expr(typecheck.NodAddrAt(pos, x)).(*ir.AddrExpr)
	typed(types.NewPtr(x.Type()), n)
	return n
}

func Assert(pos src.XPos, x ir.Node, typ *types.Type) ir.Node {
	return typed(typ, ir.NewTypeAssertExpr(pos, x, nil))
}

func Binary(pos src.XPos, op ir.Op, x, y ir.Node) ir.Node {
	switch op {
	case ir.OANDAND, ir.OOROR:
		return typed(x.Type(), ir.NewLogicalExpr(pos, op, x, y))
	case ir.OADD:
		if x.Type().IsString() {
			// TODO(mdempsky): Construct OADDSTR directly.
			return typecheck.Expr(ir.NewBinaryExpr(pos, op, x, y))
		}
		fallthrough
	default:
		return typed(x.Type(), ir.NewBinaryExpr(pos, op, x, y))
	}
}

func Call(pos src.XPos, typ *types.Type, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	// TODO(mdempsky): This should not be so difficult.
	if fun.Op() == ir.OTYPE {
		// Actually a type conversion, not a function call.
		n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
		if fun.Type().Kind() == types.TTYPEPARAM {
			// For type params, don't typecheck until we actually know
			// the type.
			return typed(typ, n)
		}
		return typecheck.Expr(n)
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
		// Call to a builtin function.
		n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
		n.IsDDD = dots
		switch fun.BuiltinOp {
		case ir.OCLOSE, ir.ODELETE, ir.OPANIC, ir.OPRINT, ir.OPRINTN:
			return typecheck.Stmt(n)
		default:
			return typecheck.Expr(n)
		}
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

	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots

	if fun.Op() == ir.OXDOT {
		if fun.(*ir.SelectorExpr).X.Type().Kind() != types.TTYPEPARAM {
			base.FatalfAt(pos, "Expecting type param receiver in %v", fun)
		}
		// For methods called in a generic function, don't do any extra
		// transformations. We will do those later when we create the
		// instantiated function and have the correct receiver type.
		typed(typ, n)
		return n
	}
	if fun.Op() != ir.OFUNCINST {
		// If no type params, still do normal typechecking, since we're
		// still missing some things done by tcCall below (mainly
		// typecheckargs and typecheckaste).
		typecheck.Call(n)
		return n
	}

	n.Use = ir.CallUseExpr
	if fun.Type().NumResults() == 0 {
		n.Use = ir.CallUseStmt
	}

	// Rewrite call node depending on use.
	switch fun.Op() {
	case ir.ODOTINTER:
		n.SetOp(ir.OCALLINTER)

	case ir.ODOTMETH:
		n.SetOp(ir.OCALLMETH)

	default:
		n.SetOp(ir.OCALLFUNC)
	}

	typed(typ, n)
	return n
}

func Compare(pos src.XPos, typ *types.Type, op ir.Op, x, y ir.Node) ir.Node {
	n := ir.NewBinaryExpr(pos, op, x, y)
	if !types.Identical(x.Type(), y.Type()) {
		// TODO(mdempsky): Handle subtleties of constructing mixed-typed comparisons.
		n = typecheck.Expr(n).(*ir.BinaryExpr)
	}
	return typed(typ, n)
}

func Deref(pos src.XPos, x ir.Node) *ir.StarExpr {
	n := ir.NewStarExpr(pos, x)
	typed(x.Type().Elem(), n)
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
		return typ.Field(index)
	}
	return types.ReceiverBaseType(typ).Methods().Index(index)
}

func Index(pos src.XPos, x, index ir.Node) ir.Node {
	// TODO(mdempsky): Avoid typecheck.Expr (which will call tcIndex)
	return typecheck.Expr(ir.NewIndexExpr(pos, x, index))
}

func Slice(pos src.XPos, x, low, high, max ir.Node) ir.Node {
	op := ir.OSLICE
	if max != nil {
		op = ir.OSLICE3
	}
	// TODO(mdempsky): Avoid typecheck.Expr.
	return typecheck.Expr(ir.NewSliceExpr(pos, op, x, low, high, max))
}

func Unary(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	switch op {
	case ir.OADDR:
		return Addr(pos, x)
	case ir.ODEREF:
		return Deref(pos, x)
	}

	typ := x.Type()
	if op == ir.ORECV {
		typ = typ.Elem()
	}
	return typed(typ, ir.NewUnaryExpr(pos, op, x))
}

// Statements

var one = constant.MakeInt64(1)

func IncDec(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	x = typecheck.AssignExpr(x)
	return ir.NewAssignOpStmt(pos, op, x, typecheck.DefaultLit(ir.NewBasicLit(pos, one), x.Type()))
}
