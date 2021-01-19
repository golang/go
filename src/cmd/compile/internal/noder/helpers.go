// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"go/constant"
)

// Helpers for constructing typed IR nodes.
//
// TODO(mdempsky): Move into their own package so they can be easily
// reused by iimport and frontend optimizations.
//
// TODO(mdempsky): Update to consistently return already typechecked
// results, rather than leaving the caller responsible for using
// typecheck.Expr or typecheck.Stmt.

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

func Call(pos src.XPos, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	// TODO(mdempsky): This should not be so difficult.

	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots

	// Actually a type conversion.
	if fun.Op() == ir.OTYPE {
		return typecheck.Expr(n)
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
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

	typecheck.Call(n)
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

func Index(pos src.XPos, x, index ir.Node) ir.Node {
	// TODO(mdempsky): Avoid typecheck.Expr.
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
	typ := x.Type()
	switch op {
	case ir.OADDR:
		// TODO(mdempsky): Avoid typecheck.Expr. Probably just need to set OPTRLIT as needed.
		return typed(types.NewPtr(typ), typecheck.Expr(typecheck.NodAddrAt(pos, x)))
	case ir.ODEREF:
		return typed(typ.Elem(), ir.NewStarExpr(pos, x))
	case ir.ORECV:
		return typed(typ.Elem(), ir.NewUnaryExpr(pos, op, x))
	default:
		return typed(typ, ir.NewUnaryExpr(pos, op, x))
	}
}

// Statements

var one = constant.MakeInt64(1)

func IncDec(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	x = typecheck.AssignExpr(x)
	return ir.NewAssignOpStmt(pos, op, x, typecheck.DefaultLit(ir.NewBasicLit(pos, one), x.Type()))
}
