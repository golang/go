// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/compile/internal/types"

// select
func typecheckselect(sel *Node) {
	var def *Node
	lno := setlineno(sel)
	typecheckslice(sel.Ninit.Slice(), Etop)
	for _, ncase := range sel.List.Slice() {
		if ncase.Op != OXCASE {
			setlineno(ncase)
			Fatalf("typecheckselect %v", ncase.Op)
		}

		if ncase.List.Len() == 0 {
			// default
			if def != nil {
				yyerrorl(ncase.Pos, "multiple defaults in select (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else if ncase.List.Len() > 1 {
			yyerrorl(ncase.Pos, "select cases cannot be lists")
		} else {
			ncase.List.SetFirst(typecheck(ncase.List.First(), Etop))
			n := ncase.List.First()
			ncase.Left = n
			ncase.List.Set(nil)
			switch n.Op {
			default:
				yyerrorl(n.Pos, "select case must be receive, send or assign recv")

			// convert x = <-c into OSELRECV(x, <-c).
			// remove implicit conversions; the eventual assignment
			// will reintroduce them.
			case OAS:
				if (n.Right.Op == OCONVNOP || n.Right.Op == OCONVIFACE) && n.Right.Implicit() {
					n.Right = n.Right.Left
				}

				if n.Right.Op != ORECV {
					yyerrorl(n.Pos, "select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV

				// convert x, ok = <-c into OSELRECV2(x, <-c) with ntest=ok
			case OAS2RECV:
				if n.Rlist.First().Op != ORECV {
					yyerrorl(n.Pos, "select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV2
				n.Left = n.List.First()
				n.List.Set1(n.List.Second())
				n.Right = n.Rlist.First()
				n.Rlist.Set(nil)

				// convert <-c into OSELRECV(N, <-c)
			case ORECV:
				n = nodl(n.Pos, OSELRECV, nil, n)

				n.SetTypecheck(1)
				ncase.Left = n

			case OSEND:
				break
			}
		}

		typecheckslice(ncase.Nbody.Slice(), Etop)
	}

	lineno = lno
}

func walkselect(sel *Node) {
	lno := setlineno(sel)
	if sel.Nbody.Len() != 0 {
		Fatalf("double walkselect")
	}

	init := sel.Ninit.Slice()
	sel.Ninit.Set(nil)

	init = append(init, walkselectcases(&sel.List)...)
	sel.List.Set(nil)

	sel.Nbody.Set(init)
	walkstmtlist(sel.Nbody.Slice())

	lineno = lno
}

func walkselectcases(cases *Nodes) []*Node {
	n := cases.Len()
	sellineno := lineno

	// optimization: zero-case select
	if n == 0 {
		return []*Node{mkcall("block", nil, nil)}
	}

	// optimization: one-case select: single op.
	// TODO(rsc): Reenable optimization once order.go can handle it.
	// golang.org/issue/7672.
	if n == 1 {
		cas := cases.First()
		setlineno(cas)
		l := cas.Ninit.Slice()
		if cas.Left != nil { // not default:
			n := cas.Left
			l = append(l, n.Ninit.Slice()...)
			n.Ninit.Set(nil)
			var ch *Node
			switch n.Op {
			default:
				Fatalf("select %v", n.Op)

				// ok already
			case OSEND:
				ch = n.Left

			case OSELRECV, OSELRECV2:
				ch = n.Right.Left
				if n.Op == OSELRECV || n.List.Len() == 0 {
					if n.Left == nil {
						n = n.Right
					} else {
						n.Op = OAS
					}
					break
				}

				if n.Left == nil {
					nblank = typecheck(nblank, Erv|Easgn)
					n.Left = nblank
				}

				n.Op = OAS2
				n.List.Prepend(n.Left)
				n.Rlist.Set1(n.Right)
				n.Right = nil
				n.Left = nil
				n.SetTypecheck(0)
				n = typecheck(n, Etop)
			}

			// if ch == nil { block() }; n;
			a := nod(OIF, nil, nil)

			a.Left = nod(OEQ, ch, nodnil())
			var ln Nodes
			ln.Set(l)
			a.Nbody.Set1(mkcall("block", nil, &ln))
			l = ln.Slice()
			a = typecheck(a, Etop)
			l = append(l, a, n)
		}

		l = append(l, cas.Nbody.Slice()...)
		l = append(l, nod(OBREAK, nil, nil))
		return l
	}

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	for _, cas := range cases.Slice() {
		setlineno(cas)
		n := cas.Left
		if n == nil {
			continue
		}
		switch n.Op {
		case OSEND:
			n.Right = nod(OADDR, n.Right, nil)
			n.Right = typecheck(n.Right, Erv)

		case OSELRECV, OSELRECV2:
			if n.Op == OSELRECV2 && n.List.Len() == 0 {
				n.Op = OSELRECV
			}
			if n.Op == OSELRECV2 {
				n.List.SetFirst(nod(OADDR, n.List.First(), nil))
				n.List.SetFirst(typecheck(n.List.First(), Erv))
			}

			if n.Left == nil {
				n.Left = nodnil()
			} else {
				n.Left = nod(OADDR, n.Left, nil)
				n.Left = typecheck(n.Left, Erv)
			}
		}
	}

	// optimization: two-case select but one is default: single non-blocking op.
	if n == 2 && (cases.First().Left == nil || cases.Second().Left == nil) {
		var cas *Node
		var dflt *Node
		if cases.First().Left == nil {
			cas = cases.Second()
			dflt = cases.First()
		} else {
			dflt = cases.Second()
			cas = cases.First()
		}

		n := cas.Left
		setlineno(n)
		r := nod(OIF, nil, nil)
		r.Ninit.Set(cas.Ninit.Slice())
		switch n.Op {
		default:
			Fatalf("select %v", n.Op)

		case OSEND:
			// if selectnbsend(c, v) { body } else { default body }
			ch := n.Left
			r.Left = mkcall1(chanfn("selectnbsend", 2, ch.Type), types.Types[TBOOL], &r.Ninit, ch, n.Right)

		case OSELRECV:
			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
			r = nod(OIF, nil, nil)
			r.Ninit.Set(cas.Ninit.Slice())
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv", 2, ch.Type), types.Types[TBOOL], &r.Ninit, n.Left, ch)

		case OSELRECV2:
			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
			r = nod(OIF, nil, nil)
			r.Ninit.Set(cas.Ninit.Slice())
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv2", 2, ch.Type), types.Types[TBOOL], &r.Ninit, n.Left, n.List.First(), ch)
		}

		r.Left = typecheck(r.Left, Erv)
		r.Nbody.Set(cas.Nbody.Slice())
		r.Rlist.Set(append(dflt.Ninit.Slice(), dflt.Nbody.Slice()...))
		return []*Node{r, nod(OBREAK, nil, nil)}
	}

	var init []*Node

	// generate sel-struct
	lineno = sellineno
	selv := temp(selecttype(int64(n)))
	r := nod(OAS, selv, nil)
	r = typecheck(r, Etop)
	init = append(init, r)
	var_ := conv(conv(nod(OADDR, selv, nil), types.Types[TUNSAFEPTR]), types.NewPtr(types.Types[TUINT8]))
	r = mkcall("newselect", nil, nil, var_, nodintconst(selv.Type.Width), nodintconst(int64(n)))
	r = typecheck(r, Etop)
	init = append(init, r)

	// register cases
	for _, cas := range cases.Slice() {
		setlineno(cas)

		init = append(init, cas.Ninit.Slice()...)
		cas.Ninit.Set(nil)

		var x *Node
		if n := cas.Left; n != nil {
			init = append(init, n.Ninit.Slice()...)

			switch n.Op {
			default:
				Fatalf("select %v", n.Op)
			case OSEND:
				// selectsend(sel *byte, hchan *chan any, elem *any)
				x = mkcall1(chanfn("selectsend", 2, n.Left.Type), nil, nil, var_, n.Left, n.Right)
			case OSELRECV:
				// selectrecv(sel *byte, hchan *chan any, elem *any, received *bool)
				x = mkcall1(chanfn("selectrecv", 2, n.Right.Left.Type), nil, nil, var_, n.Right.Left, n.Left, nodnil())
			case OSELRECV2:
				// selectrecv(sel *byte, hchan *chan any, elem *any, received *bool)
				x = mkcall1(chanfn("selectrecv", 2, n.Right.Left.Type), nil, nil, var_, n.Right.Left, n.Left, n.List.First())
			}
		} else {
			// selectdefault(sel *byte)
			x = mkcall("selectdefault", nil, nil, var_)
		}

		init = append(init, x)
	}

	// run the select
	lineno = sellineno
	chosen := temp(types.Types[TINT])
	r = nod(OAS, chosen, mkcall("selectgo", types.Types[TINT], nil, var_))
	r = typecheck(r, Etop)
	init = append(init, r)

	// selv is no longer alive after selectgo.
	init = append(init, nod(OVARKILL, selv, nil))

	// dispatch cases
	for i, cas := range cases.Slice() {
		setlineno(cas)

		cond := nod(OEQ, chosen, nodintconst(int64(i)))
		cond = typecheck(cond, Erv)

		r = nod(OIF, cond, nil)
		r.Nbody.AppendNodes(&cas.Nbody)
		r.Nbody.Append(nod(OBREAK, nil, nil))
		init = append(init, r)
	}

	return init
}

// Keep in sync with src/runtime/select.go.
func selecttype(size int64) *types.Type {
	// TODO(dvyukov): it's possible to generate Scase only once
	// and then cache; and also cache Select per size.

	scase := tostruct([]*Node{
		namedfield("elem", types.NewPtr(types.Types[TUINT8])),
		namedfield("chan", types.NewPtr(types.Types[TUINT8])),
		namedfield("pc", types.Types[TUINTPTR]),
		namedfield("kind", types.Types[TUINT16]),
		namedfield("receivedp", types.NewPtr(types.Types[TUINT8])),
		namedfield("releasetime", types.Types[TUINT64]),
	})
	scase.SetNoalg(true)

	sel := tostruct([]*Node{
		namedfield("tcase", types.Types[TUINT16]),
		namedfield("ncase", types.Types[TUINT16]),
		namedfield("pollorder", types.NewPtr(types.Types[TUINT8])),
		namedfield("lockorder", types.NewPtr(types.Types[TUINT8])),
		namedfield("scase", types.NewArray(scase, size)),
		namedfield("lockorderarr", types.NewArray(types.Types[TUINT16], size)),
		namedfield("pollorderarr", types.NewArray(types.Types[TUINT16], size)),
	})
	sel.SetNoalg(true)

	return sel
}
