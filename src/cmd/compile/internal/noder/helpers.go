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

// Values

func Const(pos src.XPos, typ *types.Type, val constant.Value) ir.Node {
	n := ir.NewBasicLit(pos, val)
	n.SetType(typ)
	return n
}

func Nil(pos src.XPos, typ *types.Type) ir.Node {
	n := ir.NewNilExpr(pos)
	n.SetType(typ)
	return n
}

// Expressions

func Assert(pos src.XPos, x ir.Node, typ *types.Type) ir.Node {
	return typecheck.Expr(ir.NewTypeAssertExpr(pos, x, ir.TypeNode(typ)))
}

func Binary(pos src.XPos, op ir.Op, x, y ir.Node) ir.Node {
	switch op {
	case ir.OANDAND, ir.OOROR:
		return ir.NewLogicalExpr(pos, op, x, y)
	default:
		return ir.NewBinaryExpr(pos, op, x, y)
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

	// We probably already typechecked fun, and typecheck probably
	// got it wrong because it didn't know the expression was
	// going to be called immediately. Correct its mistakes.
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
			fun.SetType(fun.Selection.Type)
		}
	}

	typecheck.Call(n)
	return n
}

func Compare(pos src.XPos, typ *types.Type, op ir.Op, x, y ir.Node) ir.Node {
	n := typecheck.Expr(ir.NewBinaryExpr(pos, op, x, y))
	n.SetType(typ)
	return n
}

func Index(pos src.XPos, x, index ir.Node) ir.Node {
	return ir.NewIndexExpr(pos, x, index)
}

func Slice(pos src.XPos, x, low, high, max ir.Node) ir.Node {
	op := ir.OSLICE
	if max != nil {
		op = ir.OSLICE3
	}
	return ir.NewSliceExpr(pos, op, x, low, high, max)
}

func Unary(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	switch op {
	case ir.OADDR:
		return typecheck.NodAddrAt(pos, x)
	case ir.ODEREF:
		return ir.NewStarExpr(pos, x)
	default:
		return ir.NewUnaryExpr(pos, op, x)
	}
}

// Statements

var one = constant.MakeInt64(1)

func IncDec(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	x = typecheck.AssignExpr(x)
	return ir.NewAssignOpStmt(pos, op, x, typecheck.DefaultLit(ir.NewBasicLit(pos, one), x.Type()))
}
