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
func typecheckselect(sel *ir.Node) {
	var def *ir.Node
	lno := setlineno(sel)
	typecheckslice(sel.Ninit.Slice(), ctxStmt)
	for _, ncase := range sel.List.Slice() {
		if ncase.Op != ir.OCASE {
			setlineno(ncase)
			base.Fatalf("typecheckselect %v", ncase.Op)
		}

		if ncase.List.Len() == 0 {
			// default
			if def != nil {
				base.ErrorfAt(ncase.Pos, "multiple defaults in select (first at %v)", ir.Line(def))
			} else {
				def = ncase
			}
		} else if ncase.List.Len() > 1 {
			base.ErrorfAt(ncase.Pos, "select cases cannot be lists")
		} else {
			ncase.List.SetFirst(typecheck(ncase.List.First(), ctxStmt))
			n := ncase.List.First()
			ncase.Left = n
			ncase.List.Set(nil)
			switch n.Op {
			default:
				pos := n.Pos
				if n.Op == ir.ONAME {
					// We don't have the right position for ONAME nodes (see #15459 and
					// others). Using ncase.Pos for now as it will provide the correct
					// line number (assuming the expression follows the "case" keyword
					// on the same line). This matches the approach before 1.10.
					pos = ncase.Pos
				}
				base.ErrorfAt(pos, "select case must be receive, send or assign recv")

			// convert x = <-c into OSELRECV(x, <-c).
			// remove implicit conversions; the eventual assignment
			// will reintroduce them.
			case ir.OAS:
				if (n.Right.Op == ir.OCONVNOP || n.Right.Op == ir.OCONVIFACE) && n.Right.Implicit() {
					n.Right = n.Right.Left
				}

				if n.Right.Op != ir.ORECV {
					base.ErrorfAt(n.Pos, "select assignment must have receive on right hand side")
					break
				}

				n.Op = ir.OSELRECV

				// convert x, ok = <-c into OSELRECV2(x, <-c) with ntest=ok
			case ir.OAS2RECV:
				if n.Right.Op != ir.ORECV {
					base.ErrorfAt(n.Pos, "select assignment must have receive on right hand side")
					break
				}

				n.Op = ir.OSELRECV2
				n.Left = n.List.First()
				n.List.Set1(n.List.Second())

				// convert <-c into OSELRECV(N, <-c)
			case ir.ORECV:
				n = ir.NodAt(n.Pos, ir.OSELRECV, nil, n)

				n.SetTypecheck(1)
				ncase.Left = n

			case ir.OSEND:
				break
			}
		}

		typecheckslice(ncase.Nbody.Slice(), ctxStmt)
	}

	base.Pos = lno
}

func walkselect(sel *ir.Node) {
	lno := setlineno(sel)
	if sel.Nbody.Len() != 0 {
		base.Fatalf("double walkselect")
	}

	init := sel.Ninit.Slice()
	sel.Ninit.Set(nil)

	init = append(init, walkselectcases(&sel.List)...)
	sel.List.Set(nil)

	sel.Nbody.Set(init)
	walkstmtlist(sel.Nbody.Slice())

	base.Pos = lno
}

func walkselectcases(cases *ir.Nodes) []*ir.Node {
	ncas := cases.Len()
	sellineno := base.Pos

	// optimization: zero-case select
	if ncas == 0 {
		return []*ir.Node{mkcall("block", nil, nil)}
	}

	// optimization: one-case select: single op.
	if ncas == 1 {
		cas := cases.First()
		setlineno(cas)
		l := cas.Ninit.Slice()
		if cas.Left != nil { // not default:
			n := cas.Left
			l = append(l, n.Ninit.Slice()...)
			n.Ninit.Set(nil)
			switch n.Op {
			default:
				base.Fatalf("select %v", n.Op)

			case ir.OSEND:
				// already ok

			case ir.OSELRECV, ir.OSELRECV2:
				if n.Op == ir.OSELRECV || n.List.Len() == 0 {
					if n.Left == nil {
						n = n.Right
					} else {
						n.Op = ir.OAS
					}
					break
				}

				if n.Left == nil {
					ir.BlankNode = typecheck(ir.BlankNode, ctxExpr|ctxAssign)
					n.Left = ir.BlankNode
				}

				n.Op = ir.OAS2
				n.List.Prepend(n.Left)
				n.Rlist.Set1(n.Right)
				n.Right = nil
				n.Left = nil
				n.SetTypecheck(0)
				n = typecheck(n, ctxStmt)
			}

			l = append(l, n)
		}

		l = append(l, cas.Nbody.Slice()...)
		l = append(l, ir.Nod(ir.OBREAK, nil, nil))
		return l
	}

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	var dflt *ir.Node
	for _, cas := range cases.Slice() {
		setlineno(cas)
		n := cas.Left
		if n == nil {
			dflt = cas
			continue
		}
		switch n.Op {
		case ir.OSEND:
			n.Right = ir.Nod(ir.OADDR, n.Right, nil)
			n.Right = typecheck(n.Right, ctxExpr)

		case ir.OSELRECV, ir.OSELRECV2:
			if n.Op == ir.OSELRECV2 && n.List.Len() == 0 {
				n.Op = ir.OSELRECV
			}

			if n.Left != nil {
				n.Left = ir.Nod(ir.OADDR, n.Left, nil)
				n.Left = typecheck(n.Left, ctxExpr)
			}
		}
	}

	// optimization: two-case select but one is default: single non-blocking op.
	if ncas == 2 && dflt != nil {
		cas := cases.First()
		if cas == dflt {
			cas = cases.Second()
		}

		n := cas.Left
		setlineno(n)
		r := ir.Nod(ir.OIF, nil, nil)
		r.Ninit.Set(cas.Ninit.Slice())
		switch n.Op {
		default:
			base.Fatalf("select %v", n.Op)

		case ir.OSEND:
			// if selectnbsend(c, v) { body } else { default body }
			ch := n.Left
			r.Left = mkcall1(chanfn("selectnbsend", 2, ch.Type), types.Types[types.TBOOL], &r.Ninit, ch, n.Right)

		case ir.OSELRECV:
			// if selectnbrecv(&v, c) { body } else { default body }
			ch := n.Right.Left
			elem := n.Left
			if elem == nil {
				elem = nodnil()
			}
			r.Left = mkcall1(chanfn("selectnbrecv", 2, ch.Type), types.Types[types.TBOOL], &r.Ninit, elem, ch)

		case ir.OSELRECV2:
			// if selectnbrecv2(&v, &received, c) { body } else { default body }
			ch := n.Right.Left
			elem := n.Left
			if elem == nil {
				elem = nodnil()
			}
			receivedp := ir.Nod(ir.OADDR, n.List.First(), nil)
			receivedp = typecheck(receivedp, ctxExpr)
			r.Left = mkcall1(chanfn("selectnbrecv2", 2, ch.Type), types.Types[types.TBOOL], &r.Ninit, elem, receivedp, ch)
		}

		r.Left = typecheck(r.Left, ctxExpr)
		r.Nbody.Set(cas.Nbody.Slice())
		r.Rlist.Set(append(dflt.Ninit.Slice(), dflt.Nbody.Slice()...))
		return []*ir.Node{r, ir.Nod(ir.OBREAK, nil, nil)}
	}

	if dflt != nil {
		ncas--
	}
	casorder := make([]*ir.Node, ncas)
	nsends, nrecvs := 0, 0

	var init []*ir.Node

	// generate sel-struct
	base.Pos = sellineno
	selv := temp(types.NewArray(scasetype(), int64(ncas)))
	r := ir.Nod(ir.OAS, selv, nil)
	r = typecheck(r, ctxStmt)
	init = append(init, r)

	// No initialization for order; runtime.selectgo is responsible for that.
	order := temp(types.NewArray(types.Types[types.TUINT16], 2*int64(ncas)))

	var pc0, pcs *ir.Node
	if base.Flag.Race {
		pcs = temp(types.NewArray(types.Types[types.TUINTPTR], int64(ncas)))
		pc0 = typecheck(ir.Nod(ir.OADDR, ir.Nod(ir.OINDEX, pcs, nodintconst(0)), nil), ctxExpr)
	} else {
		pc0 = nodnil()
	}

	// register cases
	for _, cas := range cases.Slice() {
		setlineno(cas)

		init = append(init, cas.Ninit.Slice()...)
		cas.Ninit.Set(nil)

		n := cas.Left
		if n == nil { // default:
			continue
		}

		var i int
		var c, elem *ir.Node
		switch n.Op {
		default:
			base.Fatalf("select %v", n.Op)
		case ir.OSEND:
			i = nsends
			nsends++
			c = n.Left
			elem = n.Right
		case ir.OSELRECV, ir.OSELRECV2:
			nrecvs++
			i = ncas - nrecvs
			c = n.Right.Left
			elem = n.Left
		}

		casorder[i] = cas

		setField := func(f string, val *ir.Node) {
			r := ir.Nod(ir.OAS, nodSym(ir.ODOT, ir.Nod(ir.OINDEX, selv, nodintconst(int64(i))), lookup(f)), val)
			r = typecheck(r, ctxStmt)
			init = append(init, r)
		}

		c = convnop(c, types.Types[types.TUNSAFEPTR])
		setField("c", c)
		if elem != nil {
			elem = convnop(elem, types.Types[types.TUNSAFEPTR])
			setField("elem", elem)
		}

		// TODO(mdempsky): There should be a cleaner way to
		// handle this.
		if base.Flag.Race {
			r = mkcall("selectsetpc", nil, nil, ir.Nod(ir.OADDR, ir.Nod(ir.OINDEX, pcs, nodintconst(int64(i))), nil))
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
	r = ir.Nod(ir.OAS2, nil, nil)
	r.List.Set2(chosen, recvOK)
	fn := syslook("selectgo")
	r.Rlist.Set1(mkcall1(fn, fn.Type.Results(), nil, bytePtrToIndex(selv, 0), bytePtrToIndex(order, 0), pc0, nodintconst(int64(nsends)), nodintconst(int64(nrecvs)), nodbool(dflt == nil)))
	r = typecheck(r, ctxStmt)
	init = append(init, r)

	// selv and order are no longer alive after selectgo.
	init = append(init, ir.Nod(ir.OVARKILL, selv, nil))
	init = append(init, ir.Nod(ir.OVARKILL, order, nil))
	if base.Flag.Race {
		init = append(init, ir.Nod(ir.OVARKILL, pcs, nil))
	}

	// dispatch cases
	dispatch := func(cond, cas *ir.Node) {
		cond = typecheck(cond, ctxExpr)
		cond = defaultlit(cond, nil)

		r := ir.Nod(ir.OIF, cond, nil)

		if n := cas.Left; n != nil && n.Op == ir.OSELRECV2 {
			x := ir.Nod(ir.OAS, n.List.First(), recvOK)
			x = typecheck(x, ctxStmt)
			r.Nbody.Append(x)
		}

		r.Nbody.AppendNodes(&cas.Nbody)
		r.Nbody.Append(ir.Nod(ir.OBREAK, nil, nil))
		init = append(init, r)
	}

	if dflt != nil {
		setlineno(dflt)
		dispatch(ir.Nod(ir.OLT, chosen, nodintconst(0)), dflt)
	}
	for i, cas := range casorder {
		setlineno(cas)
		dispatch(ir.Nod(ir.OEQ, chosen, nodintconst(int64(i))), cas)
	}

	return init
}

// bytePtrToIndex returns a Node representing "(*byte)(&n[i])".
func bytePtrToIndex(n *ir.Node, i int64) *ir.Node {
	s := ir.Nod(ir.OADDR, ir.Nod(ir.OINDEX, n, nodintconst(i)), nil)
	t := types.NewPtr(types.Types[types.TUINT8])
	return convnop(s, t)
}

var scase *types.Type

// Keep in sync with src/runtime/select.go.
func scasetype() *types.Type {
	if scase == nil {
		scase = tostruct([]*ir.Node{
			namedfield("c", types.Types[types.TUNSAFEPTR]),
			namedfield("elem", types.Types[types.TUNSAFEPTR]),
		})
		scase.SetNoalg(true)
	}
	return scase
}
