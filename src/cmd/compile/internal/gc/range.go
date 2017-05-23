// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// range
func typecheckrange(n *Node) {
	var toomany int
	var why string
	var t1 *Type
	var t2 *Type
	var v1 *Node
	var v2 *Node
	var ls []*Node

	// Typechecking order is important here:
	// 0. first typecheck range expression (slice/map/chan),
	//	it is evaluated only once and so logically it is not part of the loop.
	// 1. typcheck produced values,
	//	this part can declare new vars and so it must be typechecked before body,
	//	because body can contain a closure that captures the vars.
	// 2. decldepth++ to denote loop body.
	// 3. typecheck body.
	// 4. decldepth--.

	n.Right = typecheck(n.Right, Erv)

	t := n.Right.Type
	if t == nil {
		goto out
	}
	// delicate little dance.  see typecheckas2
	ls = n.List.Slice()
	for i1, n1 := range ls {
		if n1.Name == nil || n1.Name.Defn != n {
			ls[i1] = typecheck(ls[i1], Erv|Easgn)
		}
	}

	if t.IsPtr() && t.Elem().IsArray() {
		t = t.Elem()
	}
	n.Type = t

	toomany = 0
	switch t.Etype {
	default:
		Yyerror("cannot range over %v", Nconv(n.Right, FmtLong))
		goto out

	case TARRAY, TSLICE:
		t1 = Types[TINT]
		t2 = t.Elem()

	case TMAP:
		t1 = t.Key()
		t2 = t.Val()

	case TCHAN:
		if !t.ChanDir().CanRecv() {
			Yyerror("invalid operation: range %v (receive from send-only type %v)", n.Right, n.Right.Type)
			goto out
		}

		t1 = t.Elem()
		t2 = nil
		if n.List.Len() == 2 {
			toomany = 1
		}

	case TSTRING:
		t1 = Types[TINT]
		t2 = runetype
	}

	if n.List.Len() > 2 || toomany != 0 {
		Yyerror("too many variables in range")
	}

	v1 = nil
	if n.List.Len() != 0 {
		v1 = n.List.First()
	}
	v2 = nil
	if n.List.Len() > 1 {
		v2 = n.List.Second()
	}

	// this is not only a optimization but also a requirement in the spec.
	// "if the second iteration variable is the blank identifier, the range
	// clause is equivalent to the same clause with only the first variable
	// present."
	if isblank(v2) {
		if v1 != nil {
			n.List.Set1(v1)
		}
		v2 = nil
	}

	if v1 != nil {
		if v1.Name != nil && v1.Name.Defn == n {
			v1.Type = t1
		} else if v1.Type != nil && assignop(t1, v1.Type, &why) == 0 {
			Yyerror("cannot assign type %v to %v in range%s", t1, Nconv(v1, FmtLong), why)
		}
		checkassign(n, v1)
	}

	if v2 != nil {
		if v2.Name != nil && v2.Name.Defn == n {
			v2.Type = t2
		} else if v2.Type != nil && assignop(t2, v2.Type, &why) == 0 {
			Yyerror("cannot assign type %v to %v in range%s", t2, Nconv(v2, FmtLong), why)
		}
		checkassign(n, v2)
	}

	// second half of dance
out:
	n.Typecheck = 1
	ls = n.List.Slice()
	for i1, n1 := range ls {
		if n1.Typecheck == 0 {
			ls[i1] = typecheck(ls[i1], Erv|Easgn)
		}
	}

	decldepth++
	typecheckslice(n.Nbody.Slice(), Etop)
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
	lno := setlineno(a)
	n.Right = nil

	var v1 *Node
	if n.List.Len() != 0 {
		v1 = n.List.First()
	}
	var v2 *Node
	if n.List.Len() > 1 && !isblank(n.List.Second()) {
		v2 = n.List.Second()
	}

	// n.List has no meaning anymore, clear it
	// to avoid erroneous processing by racewalk.
	n.List.Set(nil)

	var body []*Node
	var init []*Node
	switch t.Etype {
	default:
		Fatalf("walkrange")

	case TARRAY, TSLICE:
		if memclrrange(n, v1, v2, a) {
			lineno = lno
			return
		}

		// orderstmt arranged for a copy of the array/slice variable if needed.
		ha := a

		hv1 := temp(Types[TINT])
		hn := temp(Types[TINT])
		var hp *Node

		init = append(init, Nod(OAS, hv1, nil))
		init = append(init, Nod(OAS, hn, Nod(OLEN, ha, nil)))
		if v2 != nil {
			hp = temp(Ptrto(n.Type.Elem()))
			tmp := Nod(OINDEX, ha, Nodintconst(0))
			tmp.Bounded = true
			init = append(init, Nod(OAS, hp, Nod(OADDR, tmp, nil)))
		}

		n.Left = Nod(OLT, hv1, hn)
		n.Right = Nod(OAS, hv1, Nod(OADD, hv1, Nodintconst(1)))
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = []*Node{Nod(OAS, v1, hv1)}
		} else {
			a := Nod(OAS2, nil, nil)
			a.List.Set([]*Node{v1, v2})
			a.Rlist.Set([]*Node{hv1, Nod(OIND, hp, nil)})
			body = []*Node{a}

			// Advance pointer as part of increment.
			// We used to advance the pointer before executing the loop body,
			// but doing so would make the pointer point past the end of the
			// array during the final iteration, possibly causing another unrelated
			// piece of memory not to be garbage collected until the loop finished.
			// Advancing during the increment ensures that the pointer p only points
			// pass the end of the array during the final "p++; i++; if(i >= len(x)) break;",
			// after which p is dead, so it cannot confuse the collector.
			tmp := Nod(OADD, hp, Nodintconst(t.Elem().Width))

			tmp.Type = hp.Type
			tmp.Typecheck = 1
			tmp.Right.Type = Types[Tptr]
			tmp.Right.Typecheck = 1
			a = Nod(OAS, hp, tmp)
			a = typecheck(a, Etop)
			n.Right.Ninit.Set1(a)
		}

	case TMAP:
		// orderstmt allocated the iterator for us.
		// we only use a once, so no copy needed.
		ha := a

		th := hiter(t)
		hit := prealloc[n]
		hit.Type = th
		n.Left = nil
		keysym := th.Field(0).Sym // depends on layout of iterator struct.  See reflect.go:hiter
		valsym := th.Field(1).Sym // ditto

		fn := syslook("mapiterinit")

		fn = substArgTypes(fn, t.Key(), t.Val(), th)
		init = append(init, mkcall1(fn, nil, nil, typename(t), ha, Nod(OADDR, hit, nil)))
		n.Left = Nod(ONE, NodSym(ODOT, hit, keysym), nodnil())

		fn = syslook("mapiternext")
		fn = substArgTypes(fn, th)
		n.Right = mkcall1(fn, nil, nil, Nod(OADDR, hit, nil))

		key := NodSym(ODOT, hit, keysym)
		key = Nod(OIND, key, nil)
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = []*Node{Nod(OAS, v1, key)}
		} else {
			val := NodSym(ODOT, hit, valsym)
			val = Nod(OIND, val, nil)
			a := Nod(OAS2, nil, nil)
			a.List.Set([]*Node{v1, v2})
			a.Rlist.Set([]*Node{key, val})
			body = []*Node{a}
		}

	case TCHAN:
		// orderstmt arranged for a copy of the channel variable.
		ha := a

		n.Left = nil

		hv1 := temp(t.Elem())
		hv1.Typecheck = 1
		if haspointers(t.Elem()) {
			init = append(init, Nod(OAS, hv1, nil))
		}
		hb := temp(Types[TBOOL])

		n.Left = Nod(ONE, hb, Nodbool(false))
		a := Nod(OAS2RECV, nil, nil)
		a.Typecheck = 1
		a.List.Set([]*Node{hv1, hb})
		a.Rlist.Set1(Nod(ORECV, ha, nil))
		n.Left.Ninit.Set1(a)
		if v1 == nil {
			body = nil
		} else {
			body = []*Node{Nod(OAS, v1, hv1)}
		}
		// Zero hv1. This prevents hv1 from being the sole, inaccessible
		// reference to an otherwise GC-able value during the next channel receive.
		// See issue 15281.
		body = append(body, Nod(OAS, hv1, nil))

	case TSTRING:
		// orderstmt arranged for a copy of the string variable.
		ha := a

		ohv1 := temp(Types[TINT])

		hv1 := temp(Types[TINT])
		init = append(init, Nod(OAS, hv1, nil))

		var a *Node
		var hv2 *Node
		if v2 == nil {
			a = Nod(OAS, hv1, mkcall("stringiter", Types[TINT], nil, ha, hv1))
		} else {
			hv2 = temp(runetype)
			a = Nod(OAS2, nil, nil)
			a.List.Set([]*Node{hv1, hv2})
			fn := syslook("stringiter2")
			a.Rlist.Set1(mkcall1(fn, fn.Type.Results(), nil, ha, hv1))
		}

		n.Left = Nod(ONE, hv1, Nodintconst(0))
		n.Left.Ninit.Set([]*Node{Nod(OAS, ohv1, hv1), a})

		body = nil
		if v1 != nil {
			body = []*Node{Nod(OAS, v1, ohv1)}
		}
		if v2 != nil {
			body = append(body, Nod(OAS, v2, hv2))
		}
	}

	n.Op = OFOR
	typecheckslice(init, Etop)
	n.Ninit.Append(init...)
	typecheckslice(n.Left.Ninit.Slice(), Etop)
	n.Left = typecheck(n.Left, Erv)
	n.Right = typecheck(n.Right, Etop)
	typecheckslice(body, Etop)
	n.Nbody.Set(append(body, n.Nbody.Slice()...))
	n = walkstmt(n)

	lineno = lno
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
	if n.Nbody.Len() == 0 || n.Nbody.First() == nil || n.Nbody.Len() > 1 {
		return false
	}
	stmt := n.Nbody.First() // only stmt in body
	if stmt.Op != OAS || stmt.Left.Op != OINDEX {
		return false
	}
	if !samesafeexpr(stmt.Left.Left, a) || !samesafeexpr(stmt.Left.Right, v1) {
		return false
	}
	elemsize := n.Type.Elem().Width
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

	n.Nbody.Set(nil)
	n.Left = Nod(ONE, Nod(OLEN, a, nil), Nodintconst(0))

	// hp = &a[0]
	hp := temp(Ptrto(Types[TUINT8]))

	tmp := Nod(OINDEX, a, Nodintconst(0))
	tmp.Bounded = true
	tmp = Nod(OADDR, tmp, nil)
	tmp = Nod(OCONVNOP, tmp, nil)
	tmp.Type = Ptrto(Types[TUINT8])
	n.Nbody.Append(Nod(OAS, hp, tmp))

	// hn = len(a) * sizeof(elem(a))
	hn := temp(Types[TUINTPTR])

	tmp = Nod(OLEN, a, nil)
	tmp = Nod(OMUL, tmp, Nodintconst(elemsize))
	tmp = conv(tmp, Types[TUINTPTR])
	n.Nbody.Append(Nod(OAS, hn, tmp))

	// memclr(hp, hn)
	fn := mkcall("memclr", nil, nil, hp, hn)

	n.Nbody.Append(fn)

	// i = len(a) - 1
	v1 = Nod(OAS, v1, Nod(OSUB, Nod(OLEN, a, nil), Nodintconst(1)))

	n.Nbody.Append(v1)

	n.Left = typecheck(n.Left, Erv)
	typecheckslice(n.Nbody.Slice(), Etop)
	n = walkstmt(n)
	return true
}
