// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// The result of walkStmt MUST be assigned back to n, e.g.
//
//	n.Left = walkStmt(n.Left)
func walkStmt(n ir.Node) ir.Node {
	if n == nil {
		return n
	}

	ir.SetPos(n)

	walkStmtList(n.Init())

	switch n.Op() {
	default:
		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			base.Errorf("%v is not a top level statement", n.Sym())
		} else {
			base.Errorf("%v is not a top level statement", n.Op())
		}
		ir.Dump("nottop", n)
		return n

	case ir.OAS,
		ir.OASOP,
		ir.OAS2,
		ir.OAS2DOTTYPE,
		ir.OAS2RECV,
		ir.OAS2FUNC,
		ir.OAS2MAPR,
		ir.OCLEAR,
		ir.OCLOSE,
		ir.OCOPY,
		ir.OCALLINTER,
		ir.OCALL,
		ir.OCALLFUNC,
		ir.ODELETE,
		ir.OSEND,
		ir.OPRINT,
		ir.OPRINTLN,
		ir.OPANIC,
		ir.ORECOVERFP,
		ir.OGETG:
		if n.Typecheck() == 0 {
			base.Fatalf("missing typecheck: %+v", n)
		}

		init := ir.TakeInit(n)
		n = walkExpr(n, &init)
		if n.Op() == ir.ONAME {
			// copy rewrote to a statement list and a temp for the length.
			// Throw away the temp to avoid plain values as statements.
			n = ir.NewBlockStmt(n.Pos(), init)
			init = nil
		}
		if len(init) > 0 {
			switch n.Op() {
			case ir.OAS, ir.OAS2, ir.OBLOCK:
				n.(ir.InitNode).PtrInit().Prepend(init...)

			default:
				init.Append(n)
				n = ir.NewBlockStmt(n.Pos(), init)
			}
		}
		return n

	// special case for a receive where we throw away
	// the value received.
	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		return walkRecv(n)

	case ir.OBREAK,
		ir.OCONTINUE,
		ir.OFALL,
		ir.OGOTO,
		ir.OLABEL,
		ir.OJUMPTABLE,
		ir.OINTERFACESWITCH,
		ir.ODCL,
		ir.OCHECKNIL:
		return n

	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		walkStmtList(n.List)
		return n

	case ir.OCASE:
		base.Errorf("case statement out of place")
		panic("unreachable")

	case ir.ODEFER:
		n := n.(*ir.GoDeferStmt)
		ir.CurFunc.SetHasDefer(true)
		ir.CurFunc.NumDefers++
		if ir.CurFunc.NumDefers > maxOpenDefers || n.DeferAt != nil {
			// Don't allow open-coded defers if there are more than
			// 8 defers in the function, since we use a single
			// byte to record active defers.
			// Also don't allow if we need to use deferprocat.
			ir.CurFunc.SetOpenCodedDeferDisallowed(true)
		}
		if n.Esc() != ir.EscNever {
			// If n.Esc is not EscNever, then this defer occurs in a loop,
			// so open-coded defers cannot be used in this function.
			ir.CurFunc.SetOpenCodedDeferDisallowed(true)
		}
		fallthrough
	case ir.OGO:
		n := n.(*ir.GoDeferStmt)
		return walkGoDefer(n)

	case ir.OFOR:
		n := n.(*ir.ForStmt)
		return walkFor(n)

	case ir.OIF:
		n := n.(*ir.IfStmt)
		return walkIf(n)

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		return walkReturn(n)

	case ir.OTAILCALL:
		n := n.(*ir.TailCallStmt)

		var init ir.Nodes
		call := n.Call.(*ir.CallExpr)
		call.Fun = walkExpr(call.Fun, &init)

		if len(init) > 0 {
			init.Append(n)
			return ir.NewBlockStmt(n.Pos(), init)
		}
		return n

	case ir.OINLMARK:
		n := n.(*ir.InlineMarkStmt)
		return n

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		walkSelect(n)
		return n

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		walkSwitch(n)
		return n

	case ir.ORANGE:
		n := n.(*ir.RangeStmt)
		return walkRange(n)
	}

	// No return! Each case must return (or panic),
	// to avoid confusion about what gets returned
	// in the presence of type assertions.
}

func walkStmtList(s []ir.Node) {
	for i := range s {
		s[i] = walkStmt(s[i])
	}
}

// walkFor walks an OFOR node.
func walkFor(n *ir.ForStmt) ir.Node {
	if n.Cond != nil {
		init := ir.TakeInit(n.Cond)
		walkStmtList(init)
		n.Cond = walkExpr(n.Cond, &init)
		n.Cond = ir.InitExpr(init, n.Cond)
	}

	n.Post = walkStmt(n.Post)
	walkStmtList(n.Body)
	return n
}

// validGoDeferCall reports whether call is a valid call to appear in
// a go or defer statement; that is, whether it's a regular function
// call without arguments or results.
func validGoDeferCall(call ir.Node) bool {
	if call, ok := call.(*ir.CallExpr); ok && call.Op() == ir.OCALLFUNC && len(call.KeepAlive) == 0 {
		sig := call.Fun.Type()
		return sig.NumParams()+sig.NumResults() == 0
	}
	return false
}

// walkGoDefer walks an OGO or ODEFER node.
func walkGoDefer(n *ir.GoDeferStmt) ir.Node {
	if !validGoDeferCall(n.Call) {
		base.FatalfAt(n.Pos(), "invalid %v call: %v", n.Op(), n.Call)
	}

	var init ir.Nodes

	call := n.Call.(*ir.CallExpr)
	call.Fun = walkExpr(call.Fun, &init)

	if len(init) > 0 {
		init.Append(n)
		return ir.NewBlockStmt(n.Pos(), init)
	}
	return n
}

// walkIf walks an OIF node.
func walkIf(n *ir.IfStmt) ir.Node {
	n.Cond = walkExpr(n.Cond, n.PtrInit())
	walkStmtList(n.Body)
	walkStmtList(n.Else)
	return n
}
