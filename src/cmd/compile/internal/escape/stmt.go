// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"fmt"
)

// stmt evaluates a single Go statement.
func (e *escape) stmt(n ir.Node) {
	if n == nil {
		return
	}

	lno := ir.SetPos(n)
	defer func() {
		base.Pos = lno
	}()

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v:[%d] %v stmt: %v\n", base.FmtPos(base.Pos), e.loopDepth, e.curfn, n)
	}

	e.stmts(n.Init())

	switch n.Op() {
	default:
		base.Fatalf("unexpected stmt: %v", n)

	case ir.OFALL, ir.OINLMARK:
		// nop

	case ir.OBREAK, ir.OCONTINUE, ir.OGOTO:
		// TODO(mdempsky): Handle dead code?

	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		e.stmts(n.List)

	case ir.ODCL:
		// Record loop depth at declaration.
		n := n.(*ir.Decl)
		if !ir.IsBlank(n.X) {
			e.dcl(n.X)
		}

	case ir.OLABEL:
		n := n.(*ir.LabelStmt)
		if n.Label.IsBlank() {
			break
		}
		switch e.labels[n.Label] {
		case nonlooping:
			if base.Flag.LowerM > 2 {
				fmt.Printf("%v:%v non-looping label\n", base.FmtPos(base.Pos), n)
			}
		case looping:
			if base.Flag.LowerM > 2 {
				fmt.Printf("%v: %v looping label\n", base.FmtPos(base.Pos), n)
			}
			e.loopDepth++
		default:
			base.Fatalf("label %v missing tag", n.Label)
		}
		delete(e.labels, n.Label)

	case ir.OIF:
		n := n.(*ir.IfStmt)
		e.discard(n.Cond)
		e.block(n.Body)
		e.block(n.Else)

	case ir.OCHECKNIL:
		n := n.(*ir.UnaryExpr)
		e.discard(n.X)

	case ir.OFOR:
		n := n.(*ir.ForStmt)
		base.Assert(!n.DistinctVars) // Should all be rewritten before escape analysis
		e.loopDepth++
		e.discard(n.Cond)
		e.stmt(n.Post)
		e.block(n.Body)
		e.loopDepth--

	case ir.ORANGE:
		// for Key, Value = range X { Body }
		n := n.(*ir.RangeStmt)
		base.Assert(!n.DistinctVars) // Should all be rewritten before escape analysis

		// X is evaluated outside the loop and persists until the loop
		// terminates.
		tmp := e.newLoc(nil, true)
		e.expr(tmp.asHole(), n.X)

		e.loopDepth++
		ks := e.addrs([]ir.Node{n.Key, n.Value})
		if n.X.Type().IsArray() {
			e.flow(ks[1].note(n, "range"), tmp)
		} else {
			e.flow(ks[1].deref(n, "range-deref"), tmp)
		}
		e.reassigned(ks, n)

		e.block(n.Body)
		e.loopDepth--

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)

		if guard, ok := n.Tag.(*ir.TypeSwitchGuard); ok {
			var ks []hole
			if guard.Tag != nil {
				for _, cas := range n.Cases {
					cv := cas.Var
					k := e.dcl(cv) // type switch variables have no ODCL.
					if cv.Type().HasPointers() {
						ks = append(ks, k.dotType(cv.Type(), cas, "switch case"))
					}
				}
			}
			e.expr(e.teeHole(ks...), n.Tag.(*ir.TypeSwitchGuard).X)
		} else {
			e.discard(n.Tag)
		}

		for _, cas := range n.Cases {
			e.discards(cas.List)
			e.block(cas.Body)
		}

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		for _, cas := range n.Cases {
			e.stmt(cas.Comm)
			e.block(cas.Body)
		}
	case ir.ORECV:
		// TODO(mdempsky): Consider e.discard(n.Left).
		n := n.(*ir.UnaryExpr)
		e.exprSkipInit(e.discardHole(), n) // already visited n.Ninit
	case ir.OSEND:
		n := n.(*ir.SendStmt)
		e.discard(n.Chan)
		e.assignHeap(n.Value, "send", n)

	case ir.OAS:
		n := n.(*ir.AssignStmt)
		e.assignList([]ir.Node{n.X}, []ir.Node{n.Y}, "assign", n)
	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		// TODO(mdempsky): Worry about OLSH/ORSH?
		e.assignList([]ir.Node{n.X}, []ir.Node{n.Y}, "assign", n)
	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair", n)

	case ir.OAS2DOTTYPE: // v, ok = x.(type)
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-dot-type", n)
	case ir.OAS2MAPR: // v, ok = m[k]
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-mapr", n)
	case ir.OAS2RECV, ir.OSELRECV2: // v, ok = <-ch
		n := n.(*ir.AssignListStmt)
		e.assignList(n.Lhs, n.Rhs, "assign-pair-receive", n)

	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		e.stmts(n.Rhs[0].Init())
		ks := e.addrs(n.Lhs)
		e.call(ks, n.Rhs[0])
		e.reassigned(ks, n)
	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		results := e.curfn.Type().Results()
		dsts := make([]ir.Node, len(results))
		for i, res := range results {
			dsts[i] = res.Nname.(*ir.Name)
		}
		e.assignList(dsts, n.Results, "return", n)
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER, ir.OINLCALL, ir.OCLEAR, ir.OCLOSE, ir.OCOPY, ir.ODELETE, ir.OPANIC, ir.OPRINT, ir.OPRINTLN, ir.ORECOVERFP:
		e.call(nil, n)
	case ir.OGO, ir.ODEFER:
		n := n.(*ir.GoDeferStmt)
		e.goDeferStmt(n)

	case ir.OTAILCALL:
		n := n.(*ir.TailCallStmt)
		e.call(nil, n.Call)
	}
}

func (e *escape) stmts(l ir.Nodes) {
	for _, n := range l {
		e.stmt(n)
	}
}

// block is like stmts, but preserves loopDepth.
func (e *escape) block(l ir.Nodes) {
	old := e.loopDepth
	e.stmts(l)
	e.loopDepth = old
}

func (e *escape) dcl(n *ir.Name) hole {
	if n.Curfn != e.curfn || n.IsClosureVar() {
		base.Fatalf("bad declaration of %v", n)
	}
	loc := e.oldLoc(n)
	loc.loopDepth = e.loopDepth
	return loc.asHole()
}
