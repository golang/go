// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// select
func typecheckselect(sel *Node) {
	var ncase *Node
	var n *Node

	var def *Node
	lno := int(setlineno(sel))
	count := 0
	typechecklist(sel.Ninit, Etop)
	for l := sel.List; l != nil; l = l.Next {
		count++
		ncase = l.N
		setlineno(ncase)
		if ncase.Op != OXCASE {
			Fatalf("typecheckselect %v", Oconv(int(ncase.Op), 0))
		}

		if ncase.List == nil {
			// default
			if def != nil {
				Yyerror("multiple defaults in select (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else if ncase.List.Next != nil {
			Yyerror("select cases cannot be lists")
		} else {
			n = typecheck(&ncase.List.N, Etop)
			ncase.Left = n
			ncase.List = nil
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
				if n.Rlist.N.Op != ORECV {
					Yyerror("select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV2
				n.Left = n.List.N
				n.List = list1(n.List.Next.N)
				n.Right = n.Rlist.N
				n.Rlist = nil

				// convert <-c into OSELRECV(N, <-c)
			case ORECV:
				n = Nod(OSELRECV, nil, n)

				n.Typecheck = 1
				ncase.Left = n

			case OSEND:
				break
			}
		}

		typechecklist(ncase.Nbody, Etop)
	}

	sel.Xoffset = int64(count)
	lineno = int32(lno)
}

func walkselect(sel *Node) {
	if sel.List == nil && sel.Xoffset != 0 {
		Fatalf("double walkselect") // already rewrote
	}

	lno := int(setlineno(sel))
	i := count(sel.List)

	// optimization: zero-case select
	var init *NodeList
	var r *Node
	var n *Node
	var var_ *Node
	var selv *Node
	var cas *Node
	if i == 0 {
		sel.Nbody = list1(mkcall("block", nil, nil))
		goto out
	}

	// optimization: one-case select: single op.
	// TODO(rsc): Reenable optimization once order.go can handle it.
	// golang.org/issue/7672.
	if i == 1 {
		cas := sel.List.N
		setlineno(cas)
		l := cas.Ninit
		if cas.Left != nil { // not default:
			n := cas.Left
			l = concat(l, n.Ninit)
			n.Ninit = nil
			var ch *Node
			switch n.Op {
			default:
				Fatalf("select %v", Oconv(int(n.Op), 0))

				// ok already
			case OSEND:
				ch = n.Left

			case OSELRECV, OSELRECV2:
				ch = n.Right.Left
				if n.Op == OSELRECV || n.List == nil {
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
				n.List = concat(list1(n.Left), n.List)
				n.Rlist = list1(n.Right)
				n.Right = nil
				n.Left = nil
				n.Typecheck = 0
				typecheck(&n, Etop)
			}

			// if ch == nil { block() }; n;
			a := Nod(OIF, nil, nil)

			a.Left = Nod(OEQ, ch, nodnil())
			a.Nbody = list1(mkcall("block", nil, &l))
			typecheck(&a, Etop)
			l = list(l, a)
			l = list(l, n)
		}

		l = concat(l, cas.Nbody)
		sel.Nbody = l
		goto out
	}

	// convert case value arguments to addresses.
	// this rewrite is used by both the general code and the next optimization.
	for l := sel.List; l != nil; l = l.Next {
		cas = l.N
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
			if n.Op == OSELRECV2 && n.List == nil {
				n.Op = OSELRECV
			}
			if n.Op == OSELRECV2 {
				n.List.N = Nod(OADDR, n.List.N, nil)
				typecheck(&n.List.N, Erv)
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
	if i == 2 && (sel.List.N.Left == nil || sel.List.Next.N.Left == nil) {
		var cas *Node
		var dflt *Node
		if sel.List.N.Left == nil {
			cas = sel.List.Next.N
			dflt = sel.List.N
		} else {
			dflt = sel.List.Next.N
			cas = sel.List.N
		}

		n := cas.Left
		setlineno(n)
		r := Nod(OIF, nil, nil)
		r.Ninit = cas.Ninit
		switch n.Op {
		default:
			Fatalf("select %v", Oconv(int(n.Op), 0))

			// if selectnbsend(c, v) { body } else { default body }
		case OSEND:
			ch := n.Left

			r.Left = mkcall1(chanfn("selectnbsend", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), ch, n.Right)

			// if c != nil && selectnbrecv(&v, c) { body } else { default body }
		case OSELRECV:
			r = Nod(OIF, nil, nil)

			r.Ninit = cas.Ninit
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, ch)

			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
		case OSELRECV2:
			r = Nod(OIF, nil, nil)

			r.Ninit = cas.Ninit
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv2", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, n.List.N, ch)
		}

		typecheck(&r.Left, Erv)
		r.Nbody = cas.Nbody
		r.Rlist = concat(dflt.Ninit, dflt.Nbody)
		sel.Nbody = list1(r)
		goto out
	}

	init = sel.Ninit
	sel.Ninit = nil

	// generate sel-struct
	setlineno(sel)

	selv = temp(selecttype(int32(sel.Xoffset)))
	r = Nod(OAS, selv, nil)
	typecheck(&r, Etop)
	init = list(init, r)
	var_ = conv(conv(Nod(OADDR, selv, nil), Types[TUNSAFEPTR]), Ptrto(Types[TUINT8]))
	r = mkcall("newselect", nil, nil, var_, Nodintconst(selv.Type.Width), Nodintconst(sel.Xoffset))
	typecheck(&r, Etop)
	init = list(init, r)

	// register cases
	for l := sel.List; l != nil; l = l.Next {
		cas = l.N
		setlineno(cas)
		n = cas.Left
		r = Nod(OIF, nil, nil)
		r.Ninit = cas.Ninit
		cas.Ninit = nil
		if n != nil {
			r.Ninit = concat(r.Ninit, n.Ninit)
			n.Ninit = nil
		}

		if n == nil {
			// selectdefault(sel *byte);
			r.Left = mkcall("selectdefault", Types[TBOOL], &r.Ninit, var_)
		} else {
			switch n.Op {
			default:
				Fatalf("select %v", Oconv(int(n.Op), 0))

				// selectsend(sel *byte, hchan *chan any, elem *any) (selected bool);
			case OSEND:
				r.Left = mkcall1(chanfn("selectsend", 2, n.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Left, n.Right)

				// selectrecv(sel *byte, hchan *chan any, elem *any) (selected bool);
			case OSELRECV:
				r.Left = mkcall1(chanfn("selectrecv", 2, n.Right.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Right.Left, n.Left)

				// selectrecv2(sel *byte, hchan *chan any, elem *any, received *bool) (selected bool);
			case OSELRECV2:
				r.Left = mkcall1(chanfn("selectrecv2", 2, n.Right.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Right.Left, n.Left, n.List.N)
			}
		}

		// selv is no longer alive after use.
		r.Nbody = list(r.Nbody, Nod(OVARKILL, selv, nil))

		r.Nbody = concat(r.Nbody, cas.Nbody)
		r.Nbody = list(r.Nbody, Nod(OBREAK, nil, nil))
		init = list(init, r)
	}

	// run the select
	setlineno(sel)

	init = list(init, mkcall("selectgo", nil, nil, var_))
	sel.Nbody = init

out:
	sel.List = nil
	walkstmtlist(sel.Nbody)
	lineno = int32(lno)
}

// Keep in sync with src/runtime/runtime2.go and src/runtime/select.go.
func selecttype(size int32) *Type {
	// TODO(dvyukov): it's possible to generate SudoG and Scase only once
	// and then cache; and also cache Select per size.
	sudog := Nod(OTSTRUCT, nil, nil)

	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("g")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("selectdone")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("next")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("prev")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("nrelease")), typenod(Types[TINT32])))
	sudog.List = list(sudog.List, Nod(ODCLFIELD, newname(Lookup("waitlink")), typenod(Ptrto(Types[TUINT8]))))
	typecheck(&sudog, Etype)
	sudog.Type.Noalg = true
	sudog.Type.Local = true

	scase := Nod(OTSTRUCT, nil, nil)
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("chan")), typenod(Ptrto(Types[TUINT8]))))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("pc")), typenod(Types[TUINTPTR])))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("kind")), typenod(Types[TUINT16])))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("so")), typenod(Types[TUINT16])))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("receivedp")), typenod(Ptrto(Types[TUINT8]))))
	scase.List = list(scase.List, Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	typecheck(&scase, Etype)
	scase.Type.Noalg = true
	scase.Type.Local = true

	sel := Nod(OTSTRUCT, nil, nil)
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("tcase")), typenod(Types[TUINT16])))
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("ncase")), typenod(Types[TUINT16])))
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("pollorder")), typenod(Ptrto(Types[TUINT8]))))
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("lockorder")), typenod(Ptrto(Types[TUINT8]))))
	arr := Nod(OTARRAY, Nodintconst(int64(size)), scase)
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("scase")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Ptrto(Types[TUINT8])))
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("lockorderarr")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Types[TUINT16]))
	sel.List = list(sel.List, Nod(ODCLFIELD, newname(Lookup("pollorderarr")), arr))
	typecheck(&sel, Etype)
	sel.Type.Noalg = true
	sel.Type.Local = true

	return sel.Type
}
