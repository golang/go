// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// select
func typecheckselect(sel *ir.SelectStmt) {
	var def ir.Node
	lno := setlineno(sel)
	typecheckslice(sel.Init().Slice(), ctxStmt)
	for _, ncase := range sel.List().Slice() {
		ncase := ncase.(*ir.CaseStmt)

		if ncase.List().Len() == 0 {
			// default
			if def != nil {
				base.ErrorfAt(ncase.Pos(), "multiple defaults in select (first at %v)", ir.Line(def))
			} else {
				def = ncase
			}
		} else if ncase.List().Len() > 1 {
			base.ErrorfAt(ncase.Pos(), "select cases cannot be lists")
		} else {
			ncase.List().SetFirst(typecheck(ncase.List().First(), ctxStmt))
			n := ncase.List().First()
			ncase.SetLeft(n)
			ncase.PtrList().Set(nil)
			oselrecv2 := func(dst, recv ir.Node, colas bool) {
				n := ir.NewAssignListStmt(n.Pos(), ir.OSELRECV2, nil, nil)
				n.PtrList().Set2(dst, ir.BlankNode)
				n.PtrRlist().Set1(recv)
				n.SetColas(colas)
				n.SetTypecheck(1)
				ncase.SetLeft(n)
			}
			switch n.Op() {
			default:
				pos := n.Pos()
				if n.Op() == ir.ONAME {
					// We don't have the right position for ONAME nodes (see #15459 and
					// others). Using ncase.Pos for now as it will provide the correct
					// line number (assuming the expression follows the "case" keyword
					// on the same line). This matches the approach before 1.10.
					pos = ncase.Pos()
				}
				base.ErrorfAt(pos, "select case must be receive, send or assign recv")

			case ir.OAS:
				// convert x = <-c into x, _ = <-c
				// remove implicit conversions; the eventual assignment
				// will reintroduce them.
				n := n.(*ir.AssignStmt)
				if r := n.Right(); r.Op() == ir.OCONVNOP || r.Op() == ir.OCONVIFACE {
					r := r.(*ir.ConvExpr)
					if r.Implicit() {
						n.SetRight(r.Left())
					}
				}
				if n.Right().Op() != ir.ORECV {
					base.ErrorfAt(n.Pos(), "select assignment must have receive on right hand side")
					break
				}
				oselrecv2(n.Left(), n.Right(), n.Colas())

			case ir.OAS2RECV:
				n := n.(*ir.AssignListStmt)
				if n.Rlist().First().Op() != ir.ORECV {
					base.ErrorfAt(n.Pos(), "select assignment must have receive on right hand side")
					break
				}
				n.SetOp(ir.OSELRECV2)

			case ir.ORECV:
				// convert <-c into _, _ = <-c
				n := n.(*ir.UnaryExpr)
				oselrecv2(ir.BlankNode, n, false)

			case ir.OSEND:
				break
			}
		}

		typecheckslice(ncase.Body().Slice(), ctxStmt)
	}

	base.Pos = lno
}

func walkselect(sel *ir.SelectStmt) {
	lno := setlineno(sel)
	if sel.Body().Len() != 0 {
		base.Fatalf("double walkselect")
	}

	init := sel.Init().Slice()
	sel.PtrInit().Set(nil)

	init = append(init, walkselectcases(sel.List())...)
	sel.SetList(ir.Nodes{})

	sel.PtrBody().Set(init)
	walkstmtlist(sel.Body().Slice())

	base.Pos = lno
}

func walkselectcases(cases ir.Nodes) []ir.Node {
	ncas := cases.Len()
	sellineno := base.Pos

	// optimization: zero-case select
	if ncas == 0 {
		return []ir.Node{mkcall("block", nil, nil)}
	}

	// optimization: one-case select: single op.
	if ncas == 1 {
		cas := cases.First().(*ir.CaseStmt)
		setlineno(cas)
		l := cas.Init().Slice()
		if cas.Left() != nil { // not default:
			n := cas.Left()
			l = append(l, n.Init().Slice()...)
			n.PtrInit().Set(nil)
			switch n.Op() {
			default:
				base.Fatalf("select %v", n.Op())

			case ir.OSEND:
				// already ok

			case ir.OSELRECV2:
				r := n.(*ir.AssignListStmt)
				if ir.IsBlank(r.List().First()) && ir.IsBlank(r.List().Second()) {
					n = r.Rlist().First()
					break
				}
				r.SetOp(ir.OAS2RECV)
			}

			l = append(l, n)
		}

		l = append(l, cas.Body().Slice()...)
		l = append(l, ir.NewBranchStmt(base.Pos, ir.OBREAK, nil))
		return l
	}

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	var dflt *ir.CaseStmt
	for _, cas := range cases.Slice() {
		cas := cas.(*ir.CaseStmt)
		setlineno(cas)
		n := cas.Left()
		if n == nil {
			dflt = cas
			continue
		}
		switch n.Op() {
		case ir.OSEND:
			n := n.(*ir.SendStmt)
			n.SetRight(nodAddr(n.Right()))
			n.SetRight(typecheck(n.Right(), ctxExpr))

		case ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			if !ir.IsBlank(n.List().First()) {
				n.List().SetIndex(0, nodAddr(n.List().First()))
				n.List().SetIndex(0, typecheck(n.List().First(), ctxExpr))
			}
		}
	}

	// optimization: two-case select but one is default: single non-blocking op.
	if ncas == 2 && dflt != nil {
		cas := cases.First().(*ir.CaseStmt)
		if cas == dflt {
			cas = cases.Second().(*ir.CaseStmt)
		}

		n := cas.Left()
		setlineno(n)
		r := ir.NewIfStmt(base.Pos, nil, nil, nil)
		r.PtrInit().Set(cas.Init().Slice())
		var call ir.Node
		switch n.Op() {
		default:
			base.Fatalf("select %v", n.Op())

		case ir.OSEND:
			// if selectnbsend(c, v) { body } else { default body }
			n := n.(*ir.SendStmt)
			ch := n.Left()
			call = mkcall1(chanfn("selectnbsend", 2, ch.Type()), types.Types[types.TBOOL], r.PtrInit(), ch, n.Right())

		case ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			recv := n.Rlist().First().(*ir.UnaryExpr)
			ch := recv.Left()
			elem := n.List().First()
			if ir.IsBlank(elem) {
				elem = nodnil()
			}
			if ir.IsBlank(n.List().Second()) {
				// if selectnbrecv(&v, c) { body } else { default body }
				call = mkcall1(chanfn("selectnbrecv", 2, ch.Type()), types.Types[types.TBOOL], r.PtrInit(), elem, ch)
			} else {
				// TODO(cuonglm): make this use selectnbrecv()
				// if selectnbrecv2(&v, &received, c) { body } else { default body }
				receivedp := typecheck(nodAddr(n.List().Second()), ctxExpr)
				call = mkcall1(chanfn("selectnbrecv2", 2, ch.Type()), types.Types[types.TBOOL], r.PtrInit(), elem, receivedp, ch)
			}
		}

		r.SetLeft(typecheck(call, ctxExpr))
		r.PtrBody().Set(cas.Body().Slice())
		r.PtrRlist().Set(append(dflt.Init().Slice(), dflt.Body().Slice()...))
		return []ir.Node{r, ir.NewBranchStmt(base.Pos, ir.OBREAK, nil)}
	}

	if dflt != nil {
		ncas--
	}
	casorder := make([]*ir.CaseStmt, ncas)
	nsends, nrecvs := 0, 0

	var init []ir.Node

	// generate sel-struct
	base.Pos = sellineno
	selv := temp(types.NewArray(scasetype(), int64(ncas)))
	init = append(init, typecheck(ir.NewAssignStmt(base.Pos, selv, nil), ctxStmt))

	// No initialization for order; runtime.selectgo is responsible for that.
	order := temp(types.NewArray(types.Types[types.TUINT16], 2*int64(ncas)))

	var pc0, pcs ir.Node
	if base.Flag.Race {
		pcs = temp(types.NewArray(types.Types[types.TUINTPTR], int64(ncas)))
		pc0 = typecheck(nodAddr(ir.NewIndexExpr(base.Pos, pcs, nodintconst(0))), ctxExpr)
	} else {
		pc0 = nodnil()
	}

	// register cases
	for _, cas := range cases.Slice() {
		cas := cas.(*ir.CaseStmt)
		setlineno(cas)

		init = append(init, cas.Init().Slice()...)
		cas.PtrInit().Set(nil)

		n := cas.Left()
		if n == nil { // default:
			continue
		}

		var i int
		var c, elem ir.Node
		switch n.Op() {
		default:
			base.Fatalf("select %v", n.Op())
		case ir.OSEND:
			n := n.(*ir.SendStmt)
			i = nsends
			nsends++
			c = n.Left()
			elem = n.Right()
		case ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			nrecvs++
			i = ncas - nrecvs
			recv := n.Rlist().First().(*ir.UnaryExpr)
			c = recv.Left()
			elem = n.List().First()
		}

		casorder[i] = cas

		setField := func(f string, val ir.Node) {
			r := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, ir.NewIndexExpr(base.Pos, selv, nodintconst(int64(i))), lookup(f)), val)
			init = append(init, typecheck(r, ctxStmt))
		}

		c = convnop(c, types.Types[types.TUNSAFEPTR])
		setField("c", c)
		if !ir.IsBlank(elem) {
			elem = convnop(elem, types.Types[types.TUNSAFEPTR])
			setField("elem", elem)
		}

		// TODO(mdempsky): There should be a cleaner way to
		// handle this.
		if base.Flag.Race {
			r := mkcall("selectsetpc", nil, nil, nodAddr(ir.NewIndexExpr(base.Pos, pcs, nodintconst(int64(i)))))
			init = append(init, r)
		}
	}
	if nsends+nrecvs != ncas {
		base.Fatalf("walkselectcases: miscount: %v + %v != %v", nsends, nrecvs, ncas)
	}

	// run the select
	base.Pos = sellineno
	chosen := temp(types.Types[types.TINT])
	recvOK := temp(types.Types[types.TBOOL])
	r := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
	r.PtrList().Set2(chosen, recvOK)
	fn := syslook("selectgo")
	r.PtrRlist().Set1(mkcall1(fn, fn.Type().Results(), nil, bytePtrToIndex(selv, 0), bytePtrToIndex(order, 0), pc0, nodintconst(int64(nsends)), nodintconst(int64(nrecvs)), nodbool(dflt == nil)))
	init = append(init, typecheck(r, ctxStmt))

	// selv and order are no longer alive after selectgo.
	init = append(init, ir.NewUnaryExpr(base.Pos, ir.OVARKILL, selv))
	init = append(init, ir.NewUnaryExpr(base.Pos, ir.OVARKILL, order))
	if base.Flag.Race {
		init = append(init, ir.NewUnaryExpr(base.Pos, ir.OVARKILL, pcs))
	}

	// dispatch cases
	dispatch := func(cond ir.Node, cas *ir.CaseStmt) {
		cond = typecheck(cond, ctxExpr)
		cond = defaultlit(cond, nil)

		r := ir.NewIfStmt(base.Pos, cond, nil, nil)

		if n := cas.Left(); n != nil && n.Op() == ir.OSELRECV2 {
			n := n.(*ir.AssignListStmt)
			if !ir.IsBlank(n.List().Second()) {
				x := ir.NewAssignStmt(base.Pos, n.List().Second(), recvOK)
				r.PtrBody().Append(typecheck(x, ctxStmt))
			}
		}

		r.PtrBody().AppendNodes(cas.PtrBody())
		r.PtrBody().Append(ir.NewBranchStmt(base.Pos, ir.OBREAK, nil))
		init = append(init, r)
	}

	if dflt != nil {
		setlineno(dflt)
		dispatch(ir.NewBinaryExpr(base.Pos, ir.OLT, chosen, nodintconst(0)), dflt)
	}
	for i, cas := range casorder {
		setlineno(cas)
		dispatch(ir.NewBinaryExpr(base.Pos, ir.OEQ, chosen, nodintconst(int64(i))), cas)
	}

	return init
}

// bytePtrToIndex returns a Node representing "(*byte)(&n[i])".
func bytePtrToIndex(n ir.Node, i int64) ir.Node {
	s := nodAddr(ir.NewIndexExpr(base.Pos, n, nodintconst(i)))
	t := types.NewPtr(types.Types[types.TUINT8])
	return convnop(s, t)
}

var scase *types.Type

// Keep in sync with src/runtime/select.go.
func scasetype() *types.Type {
	if scase == nil {
		scase = tostruct([]*ir.Field{
			namedfield("c", types.Types[types.TUNSAFEPTR]),
			namedfield("elem", types.Types[types.TUNSAFEPTR]),
		})
		scase.SetNoalg(true)
	}
	return scase
}
