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
	for it := nodeSeqIterate(sel.List); !it.Done(); it.Next() {
		count++
		ncase = it.N()
		setlineno(ncase)
		if ncase.Op != OXCASE {
			Fatalf("typecheckselect %v", Oconv(ncase.Op, 0))
		}

		if nodeSeqLen(ncase.List) == 0 {
			// default
			if def != nil {
				Yyerror("multiple defaults in select (first at %v)", def.Line())
			} else {
				def = ncase
			}
		} else if nodeSeqLen(ncase.List) > 1 {
			Yyerror("select cases cannot be lists")
		} else {
			it2 := nodeSeqIterate(ncase.List)
			n = typecheck(it2.P(), Etop)
			ncase.Left = n
			setNodeSeq(&ncase.List, nil)
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
				if nodeSeqFirst(n.Rlist).Op != ORECV {
					Yyerror("select assignment must have receive on right hand side")
					break
				}

				n.Op = OSELRECV2
				n.Left = nodeSeqFirst(n.List)
				setNodeSeq(&n.List, []*Node{nodeSeqSecond(n.List)})
				n.Right = nodeSeqFirst(n.Rlist)
				setNodeSeq(&n.Rlist, nil)

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
	if nodeSeqLen(sel.List) == 0 && sel.Xoffset != 0 {
		Fatalf("double walkselect") // already rewrote
	}

	lno := setlineno(sel)
	i := nodeSeqLen(sel.List)

	// optimization: zero-case select
	var init []*Node
	var r *Node
	var n *Node
	var var_ *Node
	var selv *Node
	var cas *Node
	if i == 0 {
		sel.Nbody.Set([]*Node{mkcall("block", nil, nil)})
		goto out
	}

	// optimization: one-case select: single op.
	// TODO(rsc): Reenable optimization once order.go can handle it.
	// golang.org/issue/7672.
	if i == 1 {
		cas := nodeSeqFirst(sel.List)
		setlineno(cas)
		l := nodeSeqSlice(cas.Ninit)
		if cas.Left != nil { // not default:
			n := cas.Left
			l = append(l, nodeSeqSlice(n.Ninit)...)
			setNodeSeq(&n.Ninit, nil)
			var ch *Node
			switch n.Op {
			default:
				Fatalf("select %v", Oconv(n.Op, 0))

				// ok already
			case OSEND:
				ch = n.Left

			case OSELRECV, OSELRECV2:
				ch = n.Right.Left
				if n.Op == OSELRECV || nodeSeqLen(n.List) == 0 {
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
				setNodeSeq(&n.List, append([]*Node{n.Left}, nodeSeqSlice(n.List)...))
				setNodeSeq(&n.Rlist, []*Node{n.Right})
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
			a.Nbody.Set([]*Node{mkcall("block", nil, &ln)})
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
	for it := nodeSeqIterate(sel.List); !it.Done(); it.Next() {
		cas = it.N()
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
			if n.Op == OSELRECV2 && nodeSeqLen(n.List) == 0 {
				n.Op = OSELRECV
			}
			if n.Op == OSELRECV2 {
				it := nodeSeqIterate(n.List)
				*it.P() = Nod(OADDR, it.N(), nil)
				typecheck(it.P(), Erv)
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
	if i == 2 && (nodeSeqFirst(sel.List).Left == nil || nodeSeqSecond(sel.List).Left == nil) {
		var cas *Node
		var dflt *Node
		if nodeSeqFirst(sel.List).Left == nil {
			cas = nodeSeqSecond(sel.List)
			dflt = nodeSeqFirst(sel.List)
		} else {
			dflt = nodeSeqSecond(sel.List)
			cas = nodeSeqFirst(sel.List.Slice())
		}

		n := cas.Left
		setlineno(n)
		r := Nod(OIF, nil, nil)
		setNodeSeq(&r.Ninit, cas.Ninit)
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

			setNodeSeq(&r.Ninit, cas.Ninit)
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, ch)

			// if c != nil && selectnbrecv2(&v, c) { body } else { default body }
		case OSELRECV2:
			r = Nod(OIF, nil, nil)

			setNodeSeq(&r.Ninit, cas.Ninit)
			ch := n.Right.Left
			r.Left = mkcall1(chanfn("selectnbrecv2", 2, ch.Type), Types[TBOOL], &r.Ninit, typename(ch.Type), n.Left, nodeSeqFirst(n.List), ch)
		}

		typecheck(&r.Left, Erv)
		r.Nbody.Set(cas.Nbody.Slice())
		setNodeSeq(&r.Rlist, append(nodeSeqSlice(dflt.Ninit), dflt.Nbody.Slice()...))
		sel.Nbody.Set([]*Node{r})
		goto out
	}

	init = nodeSeqSlice(sel.Ninit)
	setNodeSeq(&sel.Ninit, nil)

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
	for it := nodeSeqIterate(sel.List); !it.Done(); it.Next() {
		cas = it.N()
		setlineno(cas)
		n = cas.Left
		r = Nod(OIF, nil, nil)
		setNodeSeq(&r.Ninit, cas.Ninit)
		setNodeSeq(&cas.Ninit, nil)
		if n != nil {
			appendNodeSeq(&r.Ninit, n.Ninit)
			setNodeSeq(&n.Ninit, nil)
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
				r.Left = mkcall1(chanfn("selectrecv2", 2, n.Right.Left.Type), Types[TBOOL], &r.Ninit, var_, n.Right.Left, n.Left, nodeSeqFirst(n.List))
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
	setNodeSeq(&sel.List, nil)
	walkstmtlist(sel.Nbody.Slice())
	lineno = lno
}

// Keep in sync with src/runtime/runtime2.go and src/runtime/select.go.
func selecttype(size int32) *Type {
	// TODO(dvyukov): it's possible to generate SudoG and Scase only once
	// and then cache; and also cache Select per size.
	sudog := Nod(OTSTRUCT, nil, nil)

	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("g")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("selectdone")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("next")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("prev")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("nrelease")), typenod(Types[TINT32])))
	appendNodeSeqNode(&sudog.List, Nod(ODCLFIELD, newname(Lookup("waitlink")), typenod(Ptrto(Types[TUINT8]))))
	typecheck(&sudog, Etype)
	sudog.Type.Noalg = true
	sudog.Type.Local = true

	scase := Nod(OTSTRUCT, nil, nil)
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("elem")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("chan")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("pc")), typenod(Types[TUINTPTR])))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("kind")), typenod(Types[TUINT16])))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("so")), typenod(Types[TUINT16])))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("receivedp")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&scase.List, Nod(ODCLFIELD, newname(Lookup("releasetime")), typenod(Types[TUINT64])))
	typecheck(&scase, Etype)
	scase.Type.Noalg = true
	scase.Type.Local = true

	sel := Nod(OTSTRUCT, nil, nil)
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("tcase")), typenod(Types[TUINT16])))
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("ncase")), typenod(Types[TUINT16])))
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("pollorder")), typenod(Ptrto(Types[TUINT8]))))
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("lockorder")), typenod(Ptrto(Types[TUINT8]))))
	arr := Nod(OTARRAY, Nodintconst(int64(size)), scase)
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("scase")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Ptrto(Types[TUINT8])))
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("lockorderarr")), arr))
	arr = Nod(OTARRAY, Nodintconst(int64(size)), typenod(Types[TUINT16]))
	appendNodeSeqNode(&sel.List, Nod(ODCLFIELD, newname(Lookup("pollorderarr")), arr))
	typecheck(&sel, Etype)
	sel.Type.Noalg = true
	sel.Type.Local = true

	return sel.Type
}
