// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func (g *irgen) stmts(stmts []syntax.Stmt) []ir.Node {
	var nodes []ir.Node
	for _, stmt := range stmts {
		switch s := g.stmt(stmt).(type) {
		case nil: // EmptyStmt
		case *ir.BlockStmt:
			nodes = append(nodes, s.List...)
		default:
			nodes = append(nodes, s)
		}
	}
	return nodes
}

func (g *irgen) stmt(stmt syntax.Stmt) ir.Node {
	// TODO(mdempsky): Remove dependency on typecheck.
	return typecheck.Stmt(g.stmt0(stmt))
}

func (g *irgen) stmt0(stmt syntax.Stmt) ir.Node {
	switch stmt := stmt.(type) {
	case nil, *syntax.EmptyStmt:
		return nil
	case *syntax.LabeledStmt:
		return g.labeledStmt(stmt)
	case *syntax.BlockStmt:
		return ir.NewBlockStmt(g.pos(stmt), g.blockStmt(stmt))
	case *syntax.ExprStmt:
		x := g.expr(stmt.X)
		if call, ok := x.(*ir.CallExpr); ok {
			call.Use = ir.CallUseStmt
		}
		return x
	case *syntax.SendStmt:
		return ir.NewSendStmt(g.pos(stmt), g.expr(stmt.Chan), g.expr(stmt.Value))
	case *syntax.DeclStmt:
		return ir.NewBlockStmt(g.pos(stmt), g.decls(stmt.DeclList))

	case *syntax.AssignStmt:
		if stmt.Op != 0 && stmt.Op != syntax.Def {
			op := g.op(stmt.Op, binOps[:])
			if stmt.Rhs == nil {
				return IncDec(g.pos(stmt), op, g.expr(stmt.Lhs))
			}
			return ir.NewAssignOpStmt(g.pos(stmt), op, g.expr(stmt.Lhs), g.expr(stmt.Rhs))
		}

		rhs := g.exprList(stmt.Rhs)
		if list, ok := stmt.Lhs.(*syntax.ListExpr); ok && len(list.ElemList) != 1 || len(rhs) != 1 {
			n := ir.NewAssignListStmt(g.pos(stmt), ir.OAS2, nil, nil)
			n.Def = stmt.Op == syntax.Def
			n.Lhs = g.assignList(stmt.Lhs, n, n.Def)
			n.Rhs = rhs
			return n
		}

		n := ir.NewAssignStmt(g.pos(stmt), nil, nil)
		n.Def = stmt.Op == syntax.Def
		n.X = g.assignList(stmt.Lhs, n, n.Def)[0]
		n.Y = rhs[0]
		return n

	case *syntax.BranchStmt:
		return ir.NewBranchStmt(g.pos(stmt), g.tokOp(int(stmt.Tok), branchOps[:]), g.name(stmt.Label))
	case *syntax.CallStmt:
		return ir.NewGoDeferStmt(g.pos(stmt), g.tokOp(int(stmt.Tok), callOps[:]), g.expr(stmt.Call))
	case *syntax.ReturnStmt:
		return ir.NewReturnStmt(g.pos(stmt), g.exprList(stmt.Results))
	case *syntax.IfStmt:
		return g.ifStmt(stmt)
	case *syntax.ForStmt:
		return g.forStmt(stmt)
	case *syntax.SelectStmt:
		return g.selectStmt(stmt)
	case *syntax.SwitchStmt:
		return g.switchStmt(stmt)

	default:
		g.unhandled("statement", stmt)
		panic("unreachable")
	}
}

// TODO(mdempsky): Investigate replacing with switch statements or dense arrays.

var branchOps = [...]ir.Op{
	syntax.Break:       ir.OBREAK,
	syntax.Continue:    ir.OCONTINUE,
	syntax.Fallthrough: ir.OFALL,
	syntax.Goto:        ir.OGOTO,
}

var callOps = [...]ir.Op{
	syntax.Defer: ir.ODEFER,
	syntax.Go:    ir.OGO,
}

func (g *irgen) tokOp(tok int, ops []ir.Op) ir.Op {
	// TODO(mdempsky): Validate.
	return ops[tok]
}

func (g *irgen) op(op syntax.Operator, ops []ir.Op) ir.Op {
	// TODO(mdempsky): Validate.
	return ops[op]
}

func (g *irgen) assignList(expr syntax.Expr, defn ir.InitNode, colas bool) []ir.Node {
	if !colas {
		return g.exprList(expr)
	}

	var exprs []syntax.Expr
	if list, ok := expr.(*syntax.ListExpr); ok {
		exprs = list.ElemList
	} else {
		exprs = []syntax.Expr{expr}
	}

	res := make([]ir.Node, len(exprs))
	for i, expr := range exprs {
		expr := expr.(*syntax.Name)
		if expr.Value == "_" {
			res[i] = ir.BlankNode
			continue
		}

		if obj, ok := g.info.Uses[expr]; ok {
			res[i] = g.obj(obj)
			continue
		}

		name, _ := g.def(expr)
		name.Defn = defn
		defn.PtrInit().Append(ir.NewDecl(name.Pos(), ir.ODCL, name))
		res[i] = name
	}
	return res
}

func (g *irgen) blockStmt(stmt *syntax.BlockStmt) []ir.Node {
	return g.stmts(stmt.List)
}

func (g *irgen) ifStmt(stmt *syntax.IfStmt) ir.Node {
	init := g.stmt(stmt.Init)
	n := ir.NewIfStmt(g.pos(stmt), g.expr(stmt.Cond), g.blockStmt(stmt.Then), nil)
	if stmt.Else != nil {
		e := g.stmt(stmt.Else)
		if e.Op() == ir.OBLOCK {
			e := e.(*ir.BlockStmt)
			n.Else = e.List
		} else {
			n.Else = []ir.Node{e}
		}
	}
	return g.init(init, n)
}

func (g *irgen) forStmt(stmt *syntax.ForStmt) ir.Node {
	if r, ok := stmt.Init.(*syntax.RangeClause); ok {
		n := ir.NewRangeStmt(g.pos(r), nil, nil, g.expr(r.X), nil)
		if r.Lhs != nil {
			n.Def = r.Def
			lhs := g.assignList(r.Lhs, n, n.Def)
			n.Key = lhs[0]
			if len(lhs) > 1 {
				n.Value = lhs[1]
			}
		}
		n.Body = g.blockStmt(stmt.Body)
		return n
	}

	return ir.NewForStmt(g.pos(stmt), g.stmt(stmt.Init), g.expr(stmt.Cond), g.stmt(stmt.Post), g.blockStmt(stmt.Body))
}

func (g *irgen) selectStmt(stmt *syntax.SelectStmt) ir.Node {
	body := make([]*ir.CommClause, len(stmt.Body))
	for i, clause := range stmt.Body {
		body[i] = ir.NewCommStmt(g.pos(clause), g.stmt(clause.Comm), g.stmts(clause.Body))
	}
	return ir.NewSelectStmt(g.pos(stmt), body)
}

func (g *irgen) switchStmt(stmt *syntax.SwitchStmt) ir.Node {
	pos := g.pos(stmt)
	init := g.stmt(stmt.Init)

	var expr ir.Node
	switch tag := stmt.Tag.(type) {
	case *syntax.TypeSwitchGuard:
		var ident *ir.Ident
		if tag.Lhs != nil {
			ident = ir.NewIdent(g.pos(tag.Lhs), g.name(tag.Lhs))
		}
		expr = ir.NewTypeSwitchGuard(pos, ident, g.expr(tag.X))
	default:
		expr = g.expr(tag)
	}

	body := make([]*ir.CaseClause, len(stmt.Body))
	for i, clause := range stmt.Body {
		// Check for an implicit clause variable before
		// visiting body, because it may contain function
		// literals that reference it, and then it'll be
		// associated to the wrong function.
		//
		// Also, override its position to the clause's colon, so that
		// dwarfgen can find the right scope for it later.
		// TODO(mdempsky): We should probably just store the scope
		// directly in the ir.Name.
		var cv *ir.Name
		if obj, ok := g.info.Implicits[clause]; ok {
			cv = g.obj(obj)
			cv.SetPos(g.makeXPos(clause.Colon))
		}
		body[i] = ir.NewCaseStmt(g.pos(clause), g.exprList(clause.Cases), g.stmts(clause.Body))
		body[i].Var = cv
	}

	return g.init(init, ir.NewSwitchStmt(pos, expr, body))
}

func (g *irgen) labeledStmt(label *syntax.LabeledStmt) ir.Node {
	sym := g.name(label.Label)
	lhs := ir.NewLabelStmt(g.pos(label), sym)
	ls := g.stmt(label.Stmt)

	// Attach label directly to control statement too.
	switch ls := ls.(type) {
	case *ir.ForStmt:
		ls.Label = sym
	case *ir.RangeStmt:
		ls.Label = sym
	case *ir.SelectStmt:
		ls.Label = sym
	case *ir.SwitchStmt:
		ls.Label = sym
	}

	l := []ir.Node{lhs}
	if ls != nil {
		if ls.Op() == ir.OBLOCK {
			ls := ls.(*ir.BlockStmt)
			l = append(l, ls.List...)
		} else {
			l = append(l, ls)
		}
	}
	return ir.NewBlockStmt(src.NoXPos, l)
}

func (g *irgen) init(init ir.Node, stmt ir.InitNode) ir.InitNode {
	if init != nil {
		stmt.SetInit([]ir.Node{init})
	}
	return stmt
}

func (g *irgen) name(name *syntax.Name) *types.Sym {
	if name == nil {
		return nil
	}
	return typecheck.Lookup(name.Value)
}
