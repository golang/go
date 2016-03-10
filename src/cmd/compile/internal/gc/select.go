// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// select
func typecheckselect(sel *Node) {
	var ncase *Node
	var n *Node

	var def *Node
	lno := setlineno(sel)
	count := 0
	typechecklist(sel.Ninit.Slice(), Etop)
	for _, n1 := range sel.List.Slice() {
		count++
		ncase = n1
		setlineno(ncase)
		if ncase.Op != OXCASE {
			Fatalf("typecheckselect %v", Oconv(ncase.Op, 0))
		}

		if ncase.List.Len() == 0 {
			// default
			if def != nil {
				Yyerror("multiple defaults in select (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else if ncase.List.Len() > 1 {
			Yyerror("select cases cannot be lists")
		} else {
			n = typecheck(ncase.List.Addr(0), Etop)
			ncase.Left = n
			ncase.List.Set(nil)
			setlineno(n)
			switch n.Op {
			default:
				Yyerror("select case must be receive, send or assign recv")

				// convert x = <-c into OSELRECV(x, <-c).
			// remove implicit conversions; the eventual assignment
			// will reintroduce them.
			case OAS:
				if (n.Right.Op == OCONVNOP || n.Right.Op == OCONVIFACE) && n.Right.Implicit {
					n.Right = n.Right.Left
				}

				if n.Right.Op != ORECV {
					Yyerror("select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV

				// convert x, ok = <-c into OSELRECV2(x, <-c) with ntest=ok
			case OAS2RECV:
				if n.Rlist.First().Op != ORECV {
					Yyerror("select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV2
				n.Left = n.List.First()
				n.List.Set1(n.List.Second())
				n.Right = n.Rlist.First()
				n.Rlist.Set(nil)

				// convert <-c into OSELRECV(N, <-c)
			case ORECV:
				n = Nod(OSELRECV, nil, n)

				n.Typecheck = 1
				ncase.Left = n

			case OSEND:
				break
			}
		}

		typechecklist(ncase.Nbody.Slice(), Etop)
	}

	sel.Xoffset = int64(count)
	lineno = lno
}

func walkselect(sel *Node) {
	if sel.List.Len() == 0 && sel.Xoffset != 0 {
		Fatalf("double walkselect") // already rewrote
	}

	lno := setlineno(sel)
	i := sel.List.Len()

	// optimization: zero-case select
	var init []*Node
	var r *Node
	var n *Node
	var var_ *Node
	var selv *Node
	if i == 0 {
		sel.Nbody.Set1(mkcall("block", nil, nil))
		goto out
	}

	// optimization: one-case select: single op.
	// TODO(rsc): Reenable optimization once order.go can handle it.
	// golang.org/issue/7672.
	if i == 1 {
		cas := sel.List.First()
		setlineno(cas)
		l := cas.Ninit.Slice()
		if cas.Left != nil { // not default:
			n := cas.Left
			l = append(l, n.Ninit.Slice()...)
			n.Ninit.Set(nil)
			var ch *Node
			switch n.Op {
			default:
				Fatalf("select %v", Oconv(n.Op, 0))

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
					typecheck(&nblank, Erv|Easgn)
					n.Left = nblank
				}

				n.Op = OAS2
				n.List.Set(append([]*Node{n.Left}, n.List.Slice()...))
				n.Rlist.Set1(n.Right)
				n.Right = nil
				n.Left = nil
				n.Typecheck = 0
				typecheck(&n, Etop)
			}

			// if ch == nil { block() }; n;
			a := Nod(OIF, nil, nil)

			a.Left = Nod(OEQ, ch, nodnil())
			var ln Nodes
			ln.Set(l)
			a.Nbody.Set1(mkcall("block", nil, &ln))
			l = ln.Slice()
			typecheck(&a, Etop)
			l = append(l, a)
			l = append(l, n)
		}

		l = append(l, cas.Nbody.Slice()...)
		sel.Nbody.Set(l)
		goto out
	}

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	for _, cas := range sel.List.Slice() {
		setlineno(cas)
		n = cas.Left
		if n == nil {
			continue
		}
		switch n.Op {
		case OSEND:
			n.Right = Nod(OADDR, n.Right, nil)
			typecheck(&n.Right, Erv)

		case OSELRECV, OSELRECV2:
			if n.Op == OSELRECV2 && n.List.Len() == 0 {
				n.Op = OSELRECV
			}
			if n.Op == OSELRECV2 {
				n.List.SetIndex(0, Nod(OADDR, n.List.First(), nil))
				typecheck(n.List.Addr(0), Erv)
			}

			if n.Left == nil {
				n.Left = nodnil()
			} else {
				n.Left = Nod(OADDR, n.Left, nil)
				typecheck(&n.Left, Erv)
			}
		}
	}

	// optimization: two-case select but one is default: single non-blocking op.
	if i == 2 && (sel.List.First().Left == nil || sel.List.Second().Left == nil) {
		var cas *Node
		var dflt *Node
		if sel.List.First().Left == nil {
			cas = sel.List.Second()
			dflt = sel.List.First()
		} else {
			dflt = sel.List.Second()
			cas = sel.List.First()
		}

		n := cas.Left
		setlineno(n)
		r := Nod(OIF, nil, nil)
		r.Ninit.Set(cas.Ninit.Slice())
		switch n.Op {
		default:
			Fatalf("select %v", Oconv(n.Op, 0))

			// if selectnbsend(c, v) { body } else { default body }
		case OSEND:
			ch := n.Left

			r.Left = mkcall1(chanfn("selectnbsend", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), ch, n.Right)

			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
		case OSELRECV:
			r = Nod(OIF, nil, nil)

			r.Ninit.Set(cas.Ninit.Slice())
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, ch)

			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
		case OSELRECV2:
			r = Nod(OIF, nil, nil)

			r.Ninit.Set(cas.Ninit.Slice())
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv2", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, n.List.First(), ch)
		}

		typecheck(&r.Left, Erv)
		r.Nbody.Set(cas.Nbody.Slice())
		r.Rlist.Set(append(dflt.Ninit.Slice(), dflt.Nbody.Slice()...))
		sel.Nbody.Set1(r)
		goto out
	}

	init = sel.Ninit.Slice()
	sel.Ninit.Set(nil)

	// generate sel-struct
	setlineno(sel)

	selv = temp(selecttype(int32(sel.Xoffset)))
	r = Nod(OAS, selv, nil)
	typecheck(&r, Etop)
	init = append(init, r)
	var_ = conv(conv(Nod(OADDR, selv, nil), Types[TUNSAFEPTR]), Ptrto(Types[TUINT8]))
	r = mkcall("newselect", nil, nil, var_, Nodintconst(selv.Type.Width), Nodintconst(sel.Xoffset))
	typecheck(&r, Etop)
	init = append(init, r)
	// register cases
	for _, cas := range sel.List.Slice() {
		setlineno(cas)
		n = cas.Left
		r = Nod(OIF, nil, nil)
		r.Ninit.Set(cas.Ninit.Slice())
		cas.Ninit.Set(nil)
		if n != nil {
			r.Ninit.AppendNodes(&n.Ninit)
			n.Ninit.Set(nil)
		}

		if n == nil {
			// selectdefault(sel *byte);
			r.Left = mkcall("selectdefault", Types[TBOOL], &r.Ninit, var_)
		} else {
			switch n.Op {
			default:
				Fatalf("select %v", Oconv(n.Op, 0))

				// selectsend(sel *byte, hchan *chan any, elem *any) (selected bool);
			case OSEND:
				r.Left = mkcall1(chanfn("selectsend", 2, n.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Left, n.Right)

				// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
			case OSELRECV:
				r.Left = mkcall1(chanfn("selectrecv", 2, n.Right.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Right.Left, n.Left)

				// selectrecv2(sel *byte, hchan *chan any, elem *any, received *bool) (selected bool);
			case OSELRECV2:
				r.Left = mkcall1(chanfn("selectrecv2", 2, n.Right.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Right.Left, n.Left, n.List.First())
			}
		}

		// selv is no longer alive after use.
		r.Nbody.Append(Nod(OVARKILL, selv, nil))

		r.Nbody.AppendNodes(&cas.Nbody)
		r.Nbody.Append(Nod(OBREAK, nil, nil))
		init = append(init, r)
	}

	// run the select
	setlineno(sel)

	init = append(init, mkcall("selectgo", nil, nil, var_))
	sel.Nbody.Set(init)

out:
	sel.List.Set(nil)
	walkstmtlist(sel.Nbody.Slice())
	lineno = lno
}

// Keep in sync with src/runtime/runtime2.go and src/runtime/select.go.
func selecttype(size int32) *Type {
	// TODO(dvyukov): it's possible to generate SudoG and Scase only once
	// and then cache; and also cache Select per size.
	sudog := Nod(OTSTRUCT, nil, nil)

	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("g")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("selectdone")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("next")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("prev")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("nrelease")), typenod(Types[TINT32])))
	sudog.List.Append(Nod(ODCLFIELD, newname(Lookup("waitlink")), typenod(Ptrto(Types[TUINT8]))))
	typecheck(&sudog, Etype)
	sudog.Type.Noalg = true
	sudog.Type.Local = true

	scase := Nod(OTSTRUCT, nil, nil)
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("chan")), typenod(Ptrto(Types[TUINT8]))))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("pc")), typenod(Types[TUINTPTR])))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("kind")), typenod(Types[TUINT16])))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("so")), typenod(Types[TUINT16])))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("receivedp")), typenod(Ptrto(Types[TUINT8]))))
	scase.List.Append(Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	typecheck(&scase, Etype)
	scase.Type.Noalg = true
	scase.Type.Local = true

	sel := Nod(OTSTRUCT, nil, nil)
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("tcase")), typenod(Types[TUINT16])))
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("ncase")), typenod(Types[TUINT16])))
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("pollorder")), typenod(Ptrto(Types[TUINT8]))))
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("lockorder")), typenod(Ptrto(Types[TUINT8]))))
	arr := Nod(OTARRAY, Nodintconst(int64(size)), scase)
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("scase")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Ptrto(Types[TUINT8])))
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("lockorderarr")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Types[TUINT16]))
	sel.List.Append(Nod(ODCLFIELD, newname(Lookup("pollorderarr")), arr))
	typecheck(&sel, Etype)
	sel.Type.Noalg = true
	sel.Type.Local = true

	return sel.Type
}
