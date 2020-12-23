// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
	"unicode/utf8"
)

// range
func typecheckrange(n *ir.RangeStmt) {
	// Typechecking order is important here:
	// 0. first typecheck range expression (slice/map/chan),
	//	it is evaluated only once and so logically it is not part of the loop.
	// 1. typecheck produced values,
	//	this part can declare new vars and so it must be typechecked before body,
	//	because body can contain a closure that captures the vars.
	// 2. decldepth++ to denote loop body.
	// 3. typecheck body.
	// 4. decldepth--.
	typecheckrangeExpr(n)

	// second half of dance, the first half being typecheckrangeExpr
	n.SetTypecheck(1)
	ls := n.List().Slice()
	for i1, n1 := range ls {
		if n1.Typecheck() == 0 {
			ls[i1] = typecheck(ls[i1], ctxExpr|ctxAssign)
		}
	}

	decldepth++
	typecheckslice(n.Body().Slice(), ctxStmt)
	decldepth--
}

func typecheckrangeExpr(n *ir.RangeStmt) {
	n.SetRight(typecheck(n.Right(), ctxExpr))

	t := n.Right().Type()
	if t == nil {
		return
	}
	// delicate little dance.  see typecheckas2
	ls := n.List().Slice()
	for i1, n1 := range ls {
		if !ir.DeclaredBy(n1, n) {
			ls[i1] = typecheck(ls[i1], ctxExpr|ctxAssign)
		}
	}

	if t.IsPtr() && t.Elem().IsArray() {
		t = t.Elem()
	}
	n.SetType(t)

	var t1, t2 *types.Type
	toomany := false
	switch t.Kind() {
	default:
		base.ErrorfAt(n.Pos(), "cannot range over %L", n.Right())
		return

	case types.TARRAY, types.TSLICE:
		t1 = types.Types[types.TINT]
		t2 = t.Elem()

	case types.TMAP:
		t1 = t.Key()
		t2 = t.Elem()

	case types.TCHAN:
		if !t.ChanDir().CanRecv() {
			base.ErrorfAt(n.Pos(), "invalid operation: range %v (receive from send-only type %v)", n.Right(), n.Right().Type())
			return
		}

		t1 = t.Elem()
		t2 = nil
		if n.List().Len() == 2 {
			toomany = true
		}

	case types.TSTRING:
		t1 = types.Types[types.TINT]
		t2 = types.RuneType
	}

	if n.List().Len() > 2 || toomany {
		base.ErrorfAt(n.Pos(), "too many variables in range")
	}

	var v1, v2 ir.Node
	if n.List().Len() != 0 {
		v1 = n.List().First()
	}
	if n.List().Len() > 1 {
		v2 = n.List().Second()
	}

	// this is not only an optimization but also a requirement in the spec.
	// "if the second iteration variable is the blank identifier, the range
	// clause is equivalent to the same clause with only the first variable
	// present."
	if ir.IsBlank(v2) {
		if v1 != nil {
			n.PtrList().Set1(v1)
		}
		v2 = nil
	}

	if v1 != nil {
		if ir.DeclaredBy(v1, n) {
			v1.SetType(t1)
		} else if v1.Type() != nil {
			if op, why := assignop(t1, v1.Type()); op == ir.OXXX {
				base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t1, v1, why)
			}
		}
		checkassign(n, v1)
	}

	if v2 != nil {
		if ir.DeclaredBy(v2, n) {
			v2.SetType(t2)
		} else if v2.Type() != nil {
			if op, why := assignop(t2, v2.Type()); op == ir.OXXX {
				base.ErrorfAt(n.Pos(), "cannot assign type %v to %L in range%s", t2, v2, why)
			}
		}
		checkassign(n, v2)
	}
}

func cheapComputableIndex(width int64) bool {
	switch thearch.LinkArch.Family {
	// MIPS does not have R+R addressing
	// Arm64 may lack ability to generate this code in our assembler,
	// but the architecture supports it.
	case sys.PPC64, sys.S390X:
		return width == 1
	case sys.AMD64, sys.I386, sys.ARM64, sys.ARM:
		switch width {
		case 1, 2, 4, 8:
			return true
		}
	}
	return false
}

// walkrange transforms various forms of ORANGE into
// simpler forms.  The result must be assigned back to n.
// Node n may also be modified in place, and may also be
// the returned node.
func walkrange(nrange *ir.RangeStmt) ir.Node {
	if isMapClear(nrange) {
		m := nrange.Right()
		lno := setlineno(m)
		n := mapClear(m)
		base.Pos = lno
		return n
	}

	nfor := ir.NewForStmt(nrange.Pos(), nil, nil, nil, nil)
	nfor.SetInit(nrange.Init())
	nfor.SetSym(nrange.Sym())

	// variable name conventions:
	//	ohv1, hv1, hv2: hidden (old) val 1, 2
	//	ha, hit: hidden aggregate, iterator
	//	hn, hp: hidden len, pointer
	//	hb: hidden bool
	//	a, v1, v2: not hidden aggregate, val 1, 2

	t := nrange.Type()

	a := nrange.Right()
	lno := setlineno(a)

	var v1, v2 ir.Node
	l := nrange.List().Len()
	if l > 0 {
		v1 = nrange.List().First()
	}

	if l > 1 {
		v2 = nrange.List().Second()
	}

	if ir.IsBlank(v2) {
		v2 = nil
	}

	if ir.IsBlank(v1) && v2 == nil {
		v1 = nil
	}

	if v1 == nil && v2 != nil {
		base.Fatalf("walkrange: v2 != nil while v1 == nil")
	}

	var ifGuard *ir.IfStmt

	var body []ir.Node
	var init []ir.Node
	switch t.Kind() {
	default:
		base.Fatalf("walkrange")

	case types.TARRAY, types.TSLICE:
		if nn := arrayClear(nrange, v1, v2, a); nn != nil {
			base.Pos = lno
			return nn
		}

		// order.stmt arranged for a copy of the array/slice variable if needed.
		ha := a

		hv1 := temp(types.Types[types.TINT])
		hn := temp(types.Types[types.TINT])

		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		init = append(init, ir.NewAssignStmt(base.Pos, hn, ir.NewUnaryExpr(base.Pos, ir.OLEN, ha)))

		nfor.SetLeft(ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn))
		nfor.SetRight(ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, nodintconst(1))))

		// for range ha { body }
		if v1 == nil {
			break
		}

		// for v1 := range ha { body }
		if v2 == nil {
			body = []ir.Node{ir.NewAssignStmt(base.Pos, v1, hv1)}
			break
		}

		// for v1, v2 := range ha { body }
		if cheapComputableIndex(nrange.Type().Elem().Width) {
			// v1, v2 = hv1, ha[hv1]
			tmp := ir.NewIndexExpr(base.Pos, ha, hv1)
			tmp.SetBounded(true)
			// Use OAS2 to correctly handle assignments
			// of the form "v1, a[v1] := range".
			a := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
			a.PtrList().Set2(v1, v2)
			a.PtrRlist().Set2(hv1, tmp)
			body = []ir.Node{a}
			break
		}

		// TODO(austin): OFORUNTIL is a strange beast, but is
		// necessary for expressing the control flow we need
		// while also making "break" and "continue" work. It
		// would be nice to just lower ORANGE during SSA, but
		// racewalk needs to see many of the operations
		// involved in ORANGE's implementation. If racewalk
		// moves into SSA, consider moving ORANGE into SSA and
		// eliminating OFORUNTIL.

		// TODO(austin): OFORUNTIL inhibits bounds-check
		// elimination on the index variable (see #20711).
		// Enhance the prove pass to understand this.
		ifGuard = ir.NewIfStmt(base.Pos, nil, nil, nil)
		ifGuard.SetLeft(ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn))
		nfor.SetOp(ir.OFORUNTIL)

		hp := temp(types.NewPtr(nrange.Type().Elem()))
		tmp := ir.NewIndexExpr(base.Pos, ha, nodintconst(0))
		tmp.SetBounded(true)
		init = append(init, ir.NewAssignStmt(base.Pos, hp, nodAddr(tmp)))

		// Use OAS2 to correctly handle assignments
		// of the form "v1, a[v1] := range".
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
		a.PtrList().Set2(v1, v2)
		a.PtrRlist().Set2(hv1, ir.NewStarExpr(base.Pos, hp))
		body = append(body, a)

		// Advance pointer as part of the late increment.
		//
		// This runs *after* the condition check, so we know
		// advancing the pointer is safe and won't go past the
		// end of the allocation.
		as := ir.NewAssignStmt(base.Pos, hp, addptr(hp, t.Elem().Width))
		nfor.PtrList().Set1(typecheck(as, ctxStmt))

	case types.TMAP:
		// order.stmt allocated the iterator for us.
		// we only use a once, so no copy needed.
		ha := a

		hit := nrange.Prealloc
		th := hit.Type()
		keysym := th.Field(0).Sym  // depends on layout of iterator struct.  See reflect.go:hiter
		elemsym := th.Field(1).Sym // ditto

		fn := syslook("mapiterinit")

		fn = substArgTypes(fn, t.Key(), t.Elem(), th)
		init = append(init, mkcall1(fn, nil, nil, typename(t), ha, nodAddr(hit)))
		nfor.SetLeft(ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym), nodnil()))

		fn = syslook("mapiternext")
		fn = substArgTypes(fn, th)
		nfor.SetRight(mkcall1(fn, nil, nil, nodAddr(hit)))

		key := ir.NewStarExpr(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym))
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = []ir.Node{ir.NewAssignStmt(base.Pos, v1, key)}
		} else {
			elem := ir.NewStarExpr(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, elemsym))
			a := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
			a.PtrList().Set2(v1, v2)
			a.PtrRlist().Set2(key, elem)
			body = []ir.Node{a}
		}

	case types.TCHAN:
		// order.stmt arranged for a copy of the channel variable.
		ha := a

		hv1 := temp(t.Elem())
		hv1.SetTypecheck(1)
		if t.Elem().HasPointers() {
			init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		}
		hb := temp(types.Types[types.TBOOL])

		nfor.SetLeft(ir.NewBinaryExpr(base.Pos, ir.ONE, hb, nodbool(false)))
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2RECV, nil, nil)
		a.SetTypecheck(1)
		a.PtrList().Set2(hv1, hb)
		a.PtrRlist().Set1(ir.NewUnaryExpr(base.Pos, ir.ORECV, ha))
		nfor.Left().PtrInit().Set1(a)
		if v1 == nil {
			body = nil
		} else {
			body = []ir.Node{ir.NewAssignStmt(base.Pos, v1, hv1)}
		}
		// Zero hv1. This prevents hv1 from being the sole, inaccessible
		// reference to an otherwise GC-able value during the next channel receive.
		// See issue 15281.
		body = append(body, ir.NewAssignStmt(base.Pos, hv1, nil))

	case types.TSTRING:
		// Transform string range statements like "for v1, v2 = range a" into
		//
		// ha := a
		// for hv1 := 0; hv1 < len(ha); {
		//   hv1t := hv1
		//   hv2 := rune(ha[hv1])
		//   if hv2 < utf8.RuneSelf {
		//      hv1++
		//   } else {
		//      hv2, hv1 = decoderune(ha, hv1)
		//   }
		//   v1, v2 = hv1t, hv2
		//   // original body
		// }

		// order.stmt arranged for a copy of the string variable.
		ha := a

		hv1 := temp(types.Types[types.TINT])
		hv1t := temp(types.Types[types.TINT])
		hv2 := temp(types.RuneType)

		// hv1 := 0
		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))

		// hv1 < len(ha)
		nfor.SetLeft(ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, ir.NewUnaryExpr(base.Pos, ir.OLEN, ha)))

		if v1 != nil {
			// hv1t = hv1
			body = append(body, ir.NewAssignStmt(base.Pos, hv1t, hv1))
		}

		// hv2 := rune(ha[hv1])
		nind := ir.NewIndexExpr(base.Pos, ha, hv1)
		nind.SetBounded(true)
		body = append(body, ir.NewAssignStmt(base.Pos, hv2, conv(nind, types.RuneType)))

		// if hv2 < utf8.RuneSelf
		nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nif.SetLeft(ir.NewBinaryExpr(base.Pos, ir.OLT, hv2, nodintconst(utf8.RuneSelf)))

		// hv1++
		nif.PtrBody().Set1(ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, nodintconst(1))))

		// } else {
		eif := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
		nif.PtrRlist().Set1(eif)

		// hv2, hv1 = decoderune(ha, hv1)
		eif.PtrList().Set2(hv2, hv1)
		fn := syslook("decoderune")
		eif.PtrRlist().Set1(mkcall1(fn, fn.Type().Results(), nil, ha, hv1))

		body = append(body, nif)

		if v1 != nil {
			if v2 != nil {
				// v1, v2 = hv1t, hv2
				a := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
				a.PtrList().Set2(v1, v2)
				a.PtrRlist().Set2(hv1t, hv2)
				body = append(body, a)
			} else {
				// v1 = hv1t
				body = append(body, ir.NewAssignStmt(base.Pos, v1, hv1t))
			}
		}
	}

	typecheckslice(init, ctxStmt)

	if ifGuard != nil {
		ifGuard.PtrInit().Append(init...)
		ifGuard = typecheck(ifGuard, ctxStmt).(*ir.IfStmt)
	} else {
		nfor.PtrInit().Append(init...)
	}

	typecheckslice(nfor.Left().Init().Slice(), ctxStmt)

	nfor.SetLeft(typecheck(nfor.Left(), ctxExpr))
	nfor.SetLeft(defaultlit(nfor.Left(), nil))
	nfor.SetRight(typecheck(nfor.Right(), ctxStmt))
	typecheckslice(body, ctxStmt)
	nfor.PtrBody().Append(body...)
	nfor.PtrBody().Append(nrange.Body().Slice()...)

	var n ir.Node = nfor
	if ifGuard != nil {
		ifGuard.PtrBody().Set1(n)
		n = ifGuard
	}

	n = walkstmt(n)

	base.Pos = lno
	return n
}

// isMapClear checks if n is of the form:
//
// for k := range m {
//   delete(m, k)
// }
//
// where == for keys of map m is reflexive.
func isMapClear(n *ir.RangeStmt) bool {
	if base.Flag.N != 0 || instrumenting {
		return false
	}

	if n.Op() != ir.ORANGE || n.Type().Kind() != types.TMAP || n.List().Len() != 1 {
		return false
	}

	k := n.List().First()
	if k == nil || ir.IsBlank(k) {
		return false
	}

	// Require k to be a new variable name.
	if !ir.DeclaredBy(k, n) {
		return false
	}

	if n.Body().Len() != 1 {
		return false
	}

	stmt := n.Body().First() // only stmt in body
	if stmt == nil || stmt.Op() != ir.ODELETE {
		return false
	}

	m := n.Right()
	if delete := stmt.(*ir.CallExpr); !samesafeexpr(delete.List().First(), m) || !samesafeexpr(delete.List().Second(), k) {
		return false
	}

	// Keys where equality is not reflexive can not be deleted from maps.
	if !isreflexive(m.Type().Key()) {
		return false
	}

	return true
}

// mapClear constructs a call to runtime.mapclear for the map m.
func mapClear(m ir.Node) ir.Node {
	t := m.Type()

	// instantiate mapclear(typ *type, hmap map[any]any)
	fn := syslook("mapclear")
	fn = substArgTypes(fn, t.Key(), t.Elem())
	n := mkcall1(fn, nil, nil, typename(t), m)
	return walkstmt(typecheck(n, ctxStmt))
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
func arrayClear(loop *ir.RangeStmt, v1, v2, a ir.Node) ir.Node {
	if base.Flag.N != 0 || instrumenting {
		return nil
	}

	if v1 == nil || v2 != nil {
		return nil
	}

	if loop.Body().Len() != 1 || loop.Body().First() == nil {
		return nil
	}

	stmt1 := loop.Body().First() // only stmt in body
	if stmt1.Op() != ir.OAS {
		return nil
	}
	stmt := stmt1.(*ir.AssignStmt)
	if stmt.Left().Op() != ir.OINDEX {
		return nil
	}
	lhs := stmt.Left().(*ir.IndexExpr)

	if !samesafeexpr(lhs.Left(), a) || !samesafeexpr(lhs.Right(), v1) {
		return nil
	}

	elemsize := loop.Type().Elem().Width
	if elemsize <= 0 || !isZero(stmt.Right()) {
		return nil
	}

	// Convert to
	// if len(a) != 0 {
	// 	hp = &a[0]
	// 	hn = len(a)*sizeof(elem(a))
	// 	memclr{NoHeap,Has}Pointers(hp, hn)
	// 	i = len(a) - 1
	// }
	n := ir.NewIfStmt(base.Pos, nil, nil, nil)
	n.PtrBody().Set(nil)
	n.SetLeft(ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), nodintconst(0)))

	// hp = &a[0]
	hp := temp(types.Types[types.TUNSAFEPTR])

	ix := ir.NewIndexExpr(base.Pos, a, nodintconst(0))
	ix.SetBounded(true)
	addr := convnop(nodAddr(ix), types.Types[types.TUNSAFEPTR])
	n.PtrBody().Append(ir.NewAssignStmt(base.Pos, hp, addr))

	// hn = len(a) * sizeof(elem(a))
	hn := temp(types.Types[types.TUINTPTR])
	mul := conv(ir.NewBinaryExpr(base.Pos, ir.OMUL, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), nodintconst(elemsize)), types.Types[types.TUINTPTR])
	n.PtrBody().Append(ir.NewAssignStmt(base.Pos, hn, mul))

	var fn ir.Node
	if a.Type().Elem().HasPointers() {
		// memclrHasPointers(hp, hn)
		Curfn.SetWBPos(stmt.Pos())
		fn = mkcall("memclrHasPointers", nil, nil, hp, hn)
	} else {
		// memclrNoHeapPointers(hp, hn)
		fn = mkcall("memclrNoHeapPointers", nil, nil, hp, hn)
	}

	n.PtrBody().Append(fn)

	// i = len(a) - 1
	v1 = ir.NewAssignStmt(base.Pos, v1, ir.NewBinaryExpr(base.Pos, ir.OSUB, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), nodintconst(1)))

	n.PtrBody().Append(v1)

	n.SetLeft(typecheck(n.Left(), ctxExpr))
	n.SetLeft(defaultlit(n.Left(), nil))
	typecheckslice(n.Body().Slice(), ctxStmt)
	return walkstmt(n)
}

// addptr returns (*T)(uintptr(p) + n).
func addptr(p ir.Node, n int64) ir.Node {
	t := p.Type()

	p = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, p)
	p.SetType(types.Types[types.TUINTPTR])

	p = ir.NewBinaryExpr(base.Pos, ir.OADD, p, nodintconst(n))

	p = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, p)
	p.SetType(t)

	return p
}
