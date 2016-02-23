// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

// range
func typecheckrange(n *Node) {
	var toomany int
	var why string
	var t1 *Type
	var t2 *Type
	var v1 *Node
	var v2 *Node

	// Typechecking order is important here:
	// 0. first typecheck range expression (slice/map/chan),
	//	it is evaluated only once and so logically it is not part of the loop.
	// 1. typcheck produced values,
	//	this part can declare new vars and so it must be typechecked before body,
	//	because body can contain a closure that captures the vars.
	// 2. decldepth++ to denote loop body.
	// 3. typecheck body.
	// 4. decldepth--.

	typecheck(&n.Right, Erv)

	t := n.Right.Type
	if t == nil {
		goto out
	}

	// delicate little dance.  see typecheckas2
	for ll := n.List; ll != nil; ll = ll.Next {
		if ll.N.Name == nil || ll.N.Name.Defn != n {
			typecheck(&ll.N, Erv|Easgn)
		}
	}

	if Isptr[t.Etype] && Isfixedarray(t.Type) {
		t = t.Type
	}
	n.Type = t

	toomany = 0
	switch t.Etype {
	default:
		Yyerror("cannot range over %v", Nconv(n.Right, obj.FmtLong))
		goto out

	case TARRAY:
		t1 = Types[TINT]
		t2 = t.Type

	case TMAP:
		t1 = t.Down
		t2 = t.Type

	case TCHAN:
		if t.Chan&Crecv == 0 {
			Yyerror("invalid operation: range %v (receive from send-only type %v)", n.Right, n.Right.Type)
			goto out
		}

		t1 = t.Type
		t2 = nil
		if count(n.List) == 2 {
			toomany = 1
		}

	case TSTRING:
		t1 = Types[TINT]
		t2 = runetype
	}

	if count(n.List) > 2 || toomany != 0 {
		Yyerror("too many variables in range")
	}

	v1 = nil
	if n.List != nil {
		v1 = n.List.N
	}
	v2 = nil
	if n.List != nil && n.List.Next != nil {
		v2 = n.List.Next.N
	}

	// this is not only a optimization but also a requirement in the spec.
	// "if the second iteration variable is the blank identifier, the range
	// clause is equivalent to the same clause with only the first variable
	// present."
	if isblank(v2) {
		if v1 != nil {
			n.List = list1(v1)
		}
		v2 = nil
	}

	if v1 != nil {
		if v1.Name != nil && v1.Name.Defn == n {
			v1.Type = t1
		} else if v1.Type != nil && assignop(t1, v1.Type, &why) == 0 {
			Yyerror("cannot assign type %v to %v in range%s", t1, Nconv(v1, obj.FmtLong), why)
		}
		checkassign(n, v1)
	}

	if v2 != nil {
		if v2.Name != nil && v2.Name.Defn == n {
			v2.Type = t2
		} else if v2.Type != nil && assignop(t2, v2.Type, &why) == 0 {
			Yyerror("cannot assign type %v to %v in range%s", t2, Nconv(v2, obj.FmtLong), why)
		}
		checkassign(n, v2)
	}

	// second half of dance
out:
	n.Typecheck = 1

	for ll := n.List; ll != nil; ll = ll.Next {
		if ll.N.Typecheck == 0 {
			typecheck(&ll.N, Erv|Easgn)
		}
	}

	decldepth++
	typechecklist(n.Nbody, Etop)
	decldepth--
}

func walkrange(n *Node) {
	// variable name conventions:
	//	ohv1, hv1, hv2: hidden (old) val 1, 2
	//	ha, hit: hidden aggregate, iterator
	//	hn, hp: hidden len, pointer
	//	hb: hidden bool
	//	a, v1, v2: not hidden aggregate, val 1, 2

	t := n.Type

	a := n.Right
	lno := int(setlineno(a))
	n.Right = nil

	var v1 *Node
	if n.List != nil {
		v1 = n.List.N
	}
	var v2 *Node
	if n.List != nil && n.List.Next != nil && !isblank(n.List.Next.N) {
		v2 = n.List.Next.N
	}

	// n->list has no meaning anymore, clear it
	// to avoid erroneous processing by racewalk.
	n.List = nil

	var body *NodeList
	var init *NodeList
	switch t.Etype {
	default:
		Fatalf("walkrange")

	case TARRAY:
		if memclrrange(n, v1, v2, a) {
			lineno = int32(lno)
			return
		}

		// orderstmt arranged for a copy of the array/slice variable if needed.
		ha := a

		hv1 := temp(Types[TINT])
		hn := temp(Types[TINT])
		var hp *Node

		init = list(init, Nod(OAS, hv1, nil))
		init = list(init, Nod(OAS, hn, Nod(OLEN, ha, nil)))
		if v2 != nil {
			hp = temp(Ptrto(n.Type.Type))
			tmp := Nod(OINDEX, ha, Nodintconst(0))
			tmp.Bounded = true
			init = list(init, Nod(OAS, hp, Nod(OADDR, tmp, nil)))
		}

		n.Left = Nod(OLT, hv1, hn)
		n.Right = Nod(OAS, hv1, Nod(OADD, hv1, Nodintconst(1)))
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = list1(Nod(OAS, v1, hv1))
		} else {
			a := Nod(OAS2, nil, nil)
			a.List = list(list1(v1), v2)
			a.Rlist = list(list1(hv1), Nod(OIND, hp, nil))
			body = list1(a)

			// Advance pointer as part of increment.
			// We used to advance the pointer before executing the loop body,
			// but doing so would make the pointer point past the end of the
			// array during the final iteration, possibly causing another unrelated
			// piece of memory not to be garbage collected until the loop finished.
			// Advancing during the increment ensures that the pointer p only points
			// pass the end of the array during the final "p++; i++; if(i >= len(x)) break;",
			// after which p is dead, so it cannot confuse the collector.
			tmp := Nod(OADD, hp, Nodintconst(t.Type.Width))

			tmp.Type = hp.Type
			tmp.Typecheck = 1
			tmp.Right.Type = Types[Tptr]
			tmp.Right.Typecheck = 1
			a = Nod(OAS, hp, tmp)
			typecheck(&a, Etop)
			n.Right.Ninit = list1(a)
		}

		// orderstmt allocated the iterator for us.
	// we only use a once, so no copy needed.
	case TMAP:
		ha := a

		th := hiter(t)
		hit := prealloc[n]
		hit.Type = th
		n.Left = nil
		keyname := newname(th.Type.Sym)      // depends on layout of iterator struct.  See reflect.go:hiter
		valname := newname(th.Type.Down.Sym) // ditto

		fn := syslook("mapiterinit", 1)

		substArgTypes(fn, t.Down, t.Type, th)
		init = list(init, mkcall1(fn, nil, nil, typename(t), ha, Nod(OADDR, hit, nil)))
		n.Left = Nod(ONE, Nod(ODOT, hit, keyname), nodnil())

		fn = syslook("mapiternext", 1)
		substArgTypes(fn, th)
		n.Right = mkcall1(fn, nil, nil, Nod(OADDR, hit, nil))

		key := Nod(ODOT, hit, keyname)
		key = Nod(OIND, key, nil)
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = list1(Nod(OAS, v1, key))
		} else {
			val := Nod(ODOT, hit, valname)
			val = Nod(OIND, val, nil)
			a := Nod(OAS2, nil, nil)
			a.List = list(list1(v1), v2)
			a.Rlist = list(list1(key), val)
			body = list1(a)
		}

		// orderstmt arranged for a copy of the channel variable.
	case TCHAN:
		ha := a

		n.Left = nil

		hv1 := temp(t.Type)
		hv1.Typecheck = 1
		if haspointers(t.Type) {
			init = list(init, Nod(OAS, hv1, nil))
		}
		hb := temp(Types[TBOOL])

		n.Left = Nod(ONE, hb, Nodbool(false))
		a := Nod(OAS2RECV, nil, nil)
		a.Typecheck = 1
		a.List = list(list1(hv1), hb)
		a.Rlist = list1(Nod(ORECV, ha, nil))
		n.Left.Ninit = list1(a)
		if v1 == nil {
			body = nil
		} else {
			body = list1(Nod(OAS, v1, hv1))
		}

		// orderstmt arranged for a copy of the string variable.
	case TSTRING:
		ha := a

		ohv1 := temp(Types[TINT])

		hv1 := temp(Types[TINT])
		init = list(init, Nod(OAS, hv1, nil))

		var a *Node
		var hv2 *Node
		if v2 == nil {
			a = Nod(OAS, hv1, mkcall("stringiter", Types[TINT], nil, ha, hv1))
		} else {
			hv2 = temp(runetype)
			a = Nod(OAS2, nil, nil)
			a.List = list(list1(hv1), hv2)
			fn := syslook("stringiter2", 0)
			a.Rlist = list1(mkcall1(fn, getoutargx(fn.Type), nil, ha, hv1))
		}

		n.Left = Nod(ONE, hv1, Nodintconst(0))
		n.Left.Ninit = list(list1(Nod(OAS, ohv1, hv1)), a)

		body = nil
		if v1 != nil {
			body = list1(Nod(OAS, v1, ohv1))
		}
		if v2 != nil {
			body = list(body, Nod(OAS, v2, hv2))
		}
	}

	n.Op = OFOR
	typechecklist(init, Etop)
	n.Ninit = concat(n.Ninit, init)
	typechecklist(n.Left.Ninit, Etop)
	typecheck(&n.Left, Erv)
	typecheck(&n.Right, Etop)
	typechecklist(body, Etop)
	n.Nbody = concat(body, n.Nbody)
	walkstmt(&n)

	lineno = int32(lno)
}

// Lower n into runtime·memclr if possible, for
// fast zeroing of slices and arrays (issue 5373).
// Look for instances of
//
// for i := range a {
// 	a[i] = zero
// }
//
// in which the evaluation of a is side-effect-free.
//
// Parameters are as in walkrange: "for v1, v2 = range a".
func memclrrange(n, v1, v2, a *Node) bool {
	if Debug['N'] != 0 || instrumenting {
		return false
	}
	if v1 == nil || v2 != nil {
		return false
	}
	if n.Nbody == nil || n.Nbody.N == nil || n.Nbody.Next != nil {
		return false
	}
	stmt := n.Nbody.N // only stmt in body
	if stmt.Op != OAS || stmt.Left.Op != OINDEX {
		return false
	}
	if !samesafeexpr(stmt.Left.Left, a) || !samesafeexpr(stmt.Left.Right, v1) {
		return false
	}
	elemsize := n.Type.Type.Width
	if elemsize <= 0 || !iszero(stmt.Right) {
		return false
	}

	// Convert to
	// if len(a) != 0 {
	// 	hp = &a[0]
	// 	hn = len(a)*sizeof(elem(a))
	// 	memclr(hp, hn)
	// 	i = len(a) - 1
	// }
	n.Op = OIF

	n.Nbody = nil
	n.Left = Nod(ONE, Nod(OLEN, a, nil), Nodintconst(0))

	// hp = &a[0]
	hp := temp(Ptrto(Types[TUINT8]))

	tmp := Nod(OINDEX, a, Nodintconst(0))
	tmp.Bounded = true
	tmp = Nod(OADDR, tmp, nil)
	tmp = Nod(OCONVNOP, tmp, nil)
	tmp.Type = Ptrto(Types[TUINT8])
	n.Nbody = list(n.Nbody, Nod(OAS, hp, tmp))

	// hn = len(a) * sizeof(elem(a))
	hn := temp(Types[TUINTPTR])

	tmp = Nod(OLEN, a, nil)
	tmp = Nod(OMUL, tmp, Nodintconst(elemsize))
	tmp = conv(tmp, Types[TUINTPTR])
	n.Nbody = list(n.Nbody, Nod(OAS, hn, tmp))

	// memclr(hp, hn)
	fn := mkcall("memclr", nil, nil, hp, hn)

	n.Nbody = list(n.Nbody, fn)

	// i = len(a) - 1
	v1 = Nod(OAS, v1, Nod(OSUB, Nod(OLEN, a, nil), Nodintconst(1)))

	n.Nbody = list(n.Nbody, v1)

	typecheck(&n.Left, Erv)
	typechecklist(n.Nbody, Etop)
	walkstmt(&n)
	return true
}
