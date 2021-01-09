// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
)

func (g *irgen) expr(expr syntax.Expr) ir.Node {
	// TODO(mdempsky): Change callers to not call on nil?
	if expr == nil {
		return nil
	}

	if expr == syntax.ImplicitOne {
		base.Fatalf("expr of ImplicitOne")
	}

	if expr, ok := expr.(*syntax.Name); ok && expr.Value == "_" {
		return ir.BlankNode
	}

	// TODO(mdempsky): Is there a better way to recognize and handle qualified identifiers?
	if expr, ok := expr.(*syntax.SelectorExpr); ok {
		if name, ok := expr.X.(*syntax.Name); ok {
			if _, ok := g.info.Uses[name].(*types2.PkgName); ok {
				return g.use(expr.Sel)
			}
		}
	}

	tv, ok := g.info.Types[expr]
	if !ok {
		base.FatalfAt(g.pos(expr), "missing type for %v (%T)", expr, expr)
	}
	switch {
	case tv.IsBuiltin():
		// TODO(mdempsky): Handle in CallExpr?
		return g.use(expr.(*syntax.Name))
	case tv.IsType():
		return ir.TypeNode(g.typ(tv.Type))
	case tv.IsValue(), tv.IsVoid():
		// ok
	default:
		base.FatalfAt(g.pos(expr), "unrecognized type-checker result")
	}

	// Constant expression.
	if tv.Value != nil {
		return Const(g.pos(expr), g.typ(tv.Type), tv.Value)
	}

	// TODO(mdempsky): Remove dependency on typecheck.Expr.
	n := typecheck.Expr(g.expr0(tv.Type, expr))
	if !g.match(n.Type(), tv.Type, tv.HasOk()) {
		base.FatalfAt(g.pos(expr), "expected %L to have type %v", n, tv.Type)
	}
	return n
}

func (g *irgen) expr0(typ types2.Type, expr syntax.Expr) ir.Node {
	pos := g.pos(expr)

	switch expr := expr.(type) {
	case *syntax.Name:
		if _, isNil := g.info.Uses[expr].(*types2.Nil); isNil {
			return Nil(pos, g.typ(typ))
		}
		return g.use(expr)

	case *syntax.CompositeLit:
		return g.compLit(typ, expr)
	case *syntax.FuncLit:
		return g.funcLit(typ, expr)

	case *syntax.AssertExpr:
		return Assert(pos, g.expr(expr.X), g.typeExpr(expr.Type))
	case *syntax.CallExpr:
		return Call(pos, g.expr(expr.Fun), g.exprs(expr.ArgList), expr.HasDots)
	case *syntax.IndexExpr:
		return Index(pos, g.expr(expr.X), g.expr(expr.Index))
	case *syntax.ParenExpr:
		return g.expr(expr.X) // skip parens; unneeded after parse+typecheck
	case *syntax.SelectorExpr:
		// TODO(mdempsky/danscales): Use g.info.Selections[expr]
		// to resolve field/method selection. See CL 280633.
		return ir.NewSelectorExpr(pos, ir.OXDOT, g.expr(expr.X), g.name(expr.Sel))
	case *syntax.SliceExpr:
		return Slice(pos, g.expr(expr.X), g.expr(expr.Index[0]), g.expr(expr.Index[1]), g.expr(expr.Index[2]))

	case *syntax.Operation:
		if expr.Y == nil {
			return Unary(pos, g.op(expr.Op, unOps[:]), g.expr(expr.X))
		}
		switch op := g.op(expr.Op, binOps[:]); op {
		case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
			return Compare(pos, g.typ(typ), op, g.expr(expr.X), g.expr(expr.Y))
		default:
			return Binary(pos, op, g.expr(expr.X), g.expr(expr.Y))
		}

	default:
		g.unhandled("expression", expr)
		panic("unreachable")
	}
}

func (g *irgen) exprList(expr syntax.Expr) []ir.Node {
	switch expr := expr.(type) {
	case nil:
		return nil
	case *syntax.ListExpr:
		return g.exprs(expr.ElemList)
	default:
		return []ir.Node{g.expr(expr)}
	}
}

func (g *irgen) exprs(exprs []syntax.Expr) []ir.Node {
	nodes := make([]ir.Node, len(exprs))
	for i, expr := range exprs {
		nodes[i] = g.expr(expr)
	}
	return nodes
}

func (g *irgen) compLit(typ types2.Type, lit *syntax.CompositeLit) ir.Node {
	if ptr, ok := typ.Underlying().(*types2.Pointer); ok {
		if _, isNamed := typ.(*types2.Named); isNamed {
			// TODO(mdempsky): Questionable, but this is
			// currently allowed by cmd/compile, go/types,
			// and gccgo:
			//
			//	type T *struct{}
			//	var _ = []T{{}}
			base.FatalfAt(g.pos(lit), "defined-pointer composite literal")
		}
		return ir.NewAddrExpr(g.pos(lit), g.compLit(ptr.Elem(), lit))
	}

	_, isStruct := typ.Underlying().(*types2.Struct)

	exprs := make([]ir.Node, len(lit.ElemList))
	for i, elem := range lit.ElemList {
		switch elem := elem.(type) {
		case *syntax.KeyValueExpr:
			if isStruct {
				exprs[i] = ir.NewStructKeyExpr(g.pos(elem), g.name(elem.Key.(*syntax.Name)), g.expr(elem.Value))
			} else {
				exprs[i] = ir.NewKeyExpr(g.pos(elem), g.expr(elem.Key), g.expr(elem.Value))
			}
		default:
			exprs[i] = g.expr(elem)
		}
	}

	return ir.NewCompLitExpr(g.pos(lit), ir.OCOMPLIT, ir.TypeNode(g.typ(typ)), exprs)
}

func (g *irgen) funcLit(typ types2.Type, expr *syntax.FuncLit) ir.Node {
	fn := ir.NewFunc(g.pos(expr))
	fn.SetIsHiddenClosure(ir.CurFunc != nil)

	fn.Nname = ir.NewNameAt(g.pos(expr), typecheck.ClosureName(ir.CurFunc))
	ir.MarkFunc(fn.Nname)
	fn.Nname.SetType(g.typ(typ))
	fn.Nname.Func = fn
	fn.Nname.Defn = fn

	fn.OClosure = ir.NewClosureExpr(g.pos(expr), fn)
	fn.OClosure.SetType(fn.Nname.Type())
	fn.OClosure.SetTypecheck(1)

	g.funcBody(fn, nil, expr.Type, expr.Body)

	ir.FinishCaptureNames(fn.Pos(), ir.CurFunc, fn)

	// TODO(mdempsky): ir.CaptureName should probably handle
	// copying these fields from the canonical variable.
	for _, cv := range fn.ClosureVars {
		cv.SetType(cv.Canonical().Type())
		cv.SetTypecheck(1)
		cv.SetWalkdef(1)
	}

	g.target.Decls = append(g.target.Decls, fn)

	return fn.OClosure
}

func (g *irgen) typeExpr(typ syntax.Expr) *types.Type {
	n := g.expr(typ)
	if n.Op() != ir.OTYPE {
		base.FatalfAt(g.pos(typ), "expected type: %L", n)
	}
	return n.Type()
}
