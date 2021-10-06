// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

func cheapComputableIndex(width int64) bool {
	switch ssagen.Arch.LinkArch.Family {
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

// walkRange transforms various forms of ORANGE into
// simpler forms.  The result must be assigned back to n.
// Node n may also be modified in place, and may also be
// the returned node.
func walkRange(nrange *ir.RangeStmt) ir.Node {
	if isMapClear(nrange) {
		m := nrange.X
		lno := ir.SetPos(m)
		n := mapClear(m)
		base.Pos = lno
		return n
	}

	nfor := ir.NewForStmt(nrange.Pos(), nil, nil, nil, nil)
	nfor.SetInit(nrange.Init())
	nfor.Label = nrange.Label

	// variable name conventions:
	//	ohv1, hv1, hv2: hidden (old) val 1, 2
	//	ha, hit: hidden aggregate, iterator
	//	hn, hp: hidden len, pointer
	//	hb: hidden bool
	//	a, v1, v2: not hidden aggregate, val 1, 2

	a := nrange.X
	t := typecheck.RangeExprType(a.Type())
	lno := ir.SetPos(a)

	v1, v2 := nrange.Key, nrange.Value

	if ir.IsBlank(v2) {
		v2 = nil
	}

	if ir.IsBlank(v1) && v2 == nil {
		v1 = nil
	}

	if v1 == nil && v2 != nil {
		base.Fatalf("walkRange: v2 != nil while v1 == nil")
	}

	var ifGuard *ir.IfStmt

	var body []ir.Node
	var init []ir.Node
	switch t.Kind() {
	default:
		base.Fatalf("walkRange")

	case types.TARRAY, types.TSLICE:
		if nn := arrayClear(nrange, v1, v2, a); nn != nil {
			base.Pos = lno
			return nn
		}

		// order.stmt arranged for a copy of the array/slice variable if needed.
		ha := a

		hv1 := typecheck.Temp(types.Types[types.TINT])
		hn := typecheck.Temp(types.Types[types.TINT])

		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		init = append(init, ir.NewAssignStmt(base.Pos, hn, ir.NewUnaryExpr(base.Pos, ir.OLEN, ha)))

		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn)
		nfor.Post = ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, ir.NewInt(1)))

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
		if cheapComputableIndex(t.Elem().Size()) {
			// v1, v2 = hv1, ha[hv1]
			tmp := ir.NewIndexExpr(base.Pos, ha, hv1)
			tmp.SetBounded(true)
			// Use OAS2 to correctly handle assignments
			// of the form "v1, a[v1] := range".
			a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{v1, v2}, []ir.Node{hv1, tmp})
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
		ifGuard.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn)
		nfor.SetOp(ir.OFORUNTIL)

		hp := typecheck.Temp(types.NewPtr(t.Elem()))
		tmp := ir.NewIndexExpr(base.Pos, ha, ir.NewInt(0))
		tmp.SetBounded(true)
		init = append(init, ir.NewAssignStmt(base.Pos, hp, typecheck.NodAddr(tmp)))

		// Use OAS2 to correctly handle assignments
		// of the form "v1, a[v1] := range".
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{v1, v2}, []ir.Node{hv1, ir.NewStarExpr(base.Pos, hp)})
		body = append(body, a)

		// Advance pointer as part of the late increment.
		//
		// This runs *after* the condition check, so we know
		// advancing the pointer is safe and won't go past the
		// end of the allocation.
		as := ir.NewAssignStmt(base.Pos, hp, addptr(hp, t.Elem().Size()))
		nfor.Late = []ir.Node{typecheck.Stmt(as)}

	case types.TMAP:
		// order.stmt allocated the iterator for us.
		// we only use a once, so no copy needed.
		ha := a

		hit := nrange.Prealloc
		th := hit.Type()
		// depends on layout of iterator struct.
		// See cmd/compile/internal/reflectdata/reflect.go:MapIterType
		keysym := th.Field(0).Sym
		elemsym := th.Field(1).Sym // ditto

		fn := typecheck.LookupRuntime("mapiterinit")

		fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem(), th)
		init = append(init, mkcallstmt1(fn, reflectdata.TypePtr(t), ha, typecheck.NodAddr(hit)))
		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym), typecheck.NodNil())

		fn = typecheck.LookupRuntime("mapiternext")
		fn = typecheck.SubstArgTypes(fn, th)
		nfor.Post = mkcallstmt1(fn, typecheck.NodAddr(hit))

		key := ir.NewStarExpr(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym))
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = []ir.Node{ir.NewAssignStmt(base.Pos, v1, key)}
		} else {
			elem := ir.NewStarExpr(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, elemsym))
			a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{v1, v2}, []ir.Node{key, elem})
			body = []ir.Node{a}
		}

	case types.TCHAN:
		// order.stmt arranged for a copy of the channel variable.
		ha := a

		hv1 := typecheck.Temp(t.Elem())
		hv1.SetTypecheck(1)
		if t.Elem().HasPointers() {
			init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		}
		hb := typecheck.Temp(types.Types[types.TBOOL])

		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, hb, ir.NewBool(false))
		lhs := []ir.Node{hv1, hb}
		rhs := []ir.Node{ir.NewUnaryExpr(base.Pos, ir.ORECV, ha)}
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2RECV, lhs, rhs)
		a.SetTypecheck(1)
		nfor.Cond = ir.InitExpr([]ir.Node{a}, nfor.Cond)
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

		hv1 := typecheck.Temp(types.Types[types.TINT])
		hv1t := typecheck.Temp(types.Types[types.TINT])
		hv2 := typecheck.Temp(types.RuneType)

		// hv1 := 0
		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))

		// hv1 < len(ha)
		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, ir.NewUnaryExpr(base.Pos, ir.OLEN, ha))

		if v1 != nil {
			// hv1t = hv1
			body = append(body, ir.NewAssignStmt(base.Pos, hv1t, hv1))
		}

		// hv2 := rune(ha[hv1])
		nind := ir.NewIndexExpr(base.Pos, ha, hv1)
		nind.SetBounded(true)
		body = append(body, ir.NewAssignStmt(base.Pos, hv2, typecheck.Conv(nind, types.RuneType)))

		// if hv2 < utf8.RuneSelf
		nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv2, ir.NewInt(utf8.RuneSelf))

		// hv1++
		nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, ir.NewInt(1)))}

		// } else {
		// hv2, hv1 = decoderune(ha, hv1)
		fn := typecheck.LookupRuntime("decoderune")
		call := mkcall1(fn, fn.Type().Results(), &nif.Else, ha, hv1)
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{hv2, hv1}, []ir.Node{call})
		nif.Else.Append(a)

		body = append(body, nif)

		if v1 != nil {
			if v2 != nil {
				// v1, v2 = hv1t, hv2
				a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{v1, v2}, []ir.Node{hv1t, hv2})
				body = append(body, a)
			} else {
				// v1 = hv1t
				body = append(body, ir.NewAssignStmt(base.Pos, v1, hv1t))
			}
		}
	}

	typecheck.Stmts(init)

	if ifGuard != nil {
		ifGuard.PtrInit().Append(init...)
		ifGuard = typecheck.Stmt(ifGuard).(*ir.IfStmt)
	} else {
		nfor.PtrInit().Append(init...)
	}

	typecheck.Stmts(nfor.Cond.Init())

	nfor.Cond = typecheck.Expr(nfor.Cond)
	nfor.Cond = typecheck.DefaultLit(nfor.Cond, nil)
	nfor.Post = typecheck.Stmt(nfor.Post)
	typecheck.Stmts(body)
	nfor.Body.Append(body...)
	nfor.Body.Append(nrange.Body...)

	var n ir.Node = nfor
	if ifGuard != nil {
		ifGuard.Body = []ir.Node{n}
		n = ifGuard
	}

	n = walkStmt(n)

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
	if base.Flag.N != 0 || base.Flag.Cfg.Instrumenting {
		return false
	}

	t := n.X.Type()
	if n.Op() != ir.ORANGE || t.Kind() != types.TMAP || n.Key == nil || n.Value != nil {
		return false
	}

	k := n.Key
	// Require k to be a new variable name.
	if !ir.DeclaredBy(k, n) {
		return false
	}

	if len(n.Body) != 1 {
		return false
	}

	stmt := n.Body[0] // only stmt in body
	if stmt == nil || stmt.Op() != ir.ODELETE {
		return false
	}

	m := n.X
	if delete := stmt.(*ir.CallExpr); !ir.SameSafeExpr(delete.Args[0], m) || !ir.SameSafeExpr(delete.Args[1], k) {
		return false
	}

	// Keys where equality is not reflexive can not be deleted from maps.
	if !types.IsReflexive(t.Key()) {
		return false
	}

	return true
}

// mapClear constructs a call to runtime.mapclear for the map m.
func mapClear(m ir.Node) ir.Node {
	t := m.Type()

	// instantiate mapclear(typ *type, hmap map[any]any)
	fn := typecheck.LookupRuntime("mapclear")
	fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem())
	n := mkcallstmt1(fn, reflectdata.TypePtr(t), m)
	return walkStmt(typecheck.Stmt(n))
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
// Parameters are as in walkRange: "for v1, v2 = range a".
func arrayClear(loop *ir.RangeStmt, v1, v2, a ir.Node) ir.Node {
	if base.Flag.N != 0 || base.Flag.Cfg.Instrumenting {
		return nil
	}

	if v1 == nil || v2 != nil {
		return nil
	}

	if len(loop.Body) != 1 || loop.Body[0] == nil {
		return nil
	}

	stmt1 := loop.Body[0] // only stmt in body
	if stmt1.Op() != ir.OAS {
		return nil
	}
	stmt := stmt1.(*ir.AssignStmt)
	if stmt.X.Op() != ir.OINDEX {
		return nil
	}
	lhs := stmt.X.(*ir.IndexExpr)

	if !ir.SameSafeExpr(lhs.X, a) || !ir.SameSafeExpr(lhs.Index, v1) {
		return nil
	}

	elemsize := typecheck.RangeExprType(loop.X.Type()).Elem().Size()
	if elemsize <= 0 || !ir.IsZero(stmt.Y) {
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
	n.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(0))

	// hp = &a[0]
	hp := typecheck.Temp(types.Types[types.TUNSAFEPTR])

	ix := ir.NewIndexExpr(base.Pos, a, ir.NewInt(0))
	ix.SetBounded(true)
	addr := typecheck.ConvNop(typecheck.NodAddr(ix), types.Types[types.TUNSAFEPTR])
	n.Body.Append(ir.NewAssignStmt(base.Pos, hp, addr))

	// hn = len(a) * sizeof(elem(a))
	hn := typecheck.Temp(types.Types[types.TUINTPTR])
	mul := typecheck.Conv(ir.NewBinaryExpr(base.Pos, ir.OMUL, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(elemsize)), types.Types[types.TUINTPTR])
	n.Body.Append(ir.NewAssignStmt(base.Pos, hn, mul))

	var fn ir.Node
	if a.Type().Elem().HasPointers() {
		// memclrHasPointers(hp, hn)
		ir.CurFunc.SetWBPos(stmt.Pos())
		fn = mkcallstmt("memclrHasPointers", hp, hn)
	} else {
		// memclrNoHeapPointers(hp, hn)
		fn = mkcallstmt("memclrNoHeapPointers", hp, hn)
	}

	n.Body.Append(fn)

	// i = len(a) - 1
	v1 = ir.NewAssignStmt(base.Pos, v1, ir.NewBinaryExpr(base.Pos, ir.OSUB, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(1)))

	n.Body.Append(v1)

	n.Cond = typecheck.Expr(n.Cond)
	n.Cond = typecheck.DefaultLit(n.Cond, nil)
	typecheck.Stmts(n.Body)
	return walkStmt(n)
}

// addptr returns (*T)(uintptr(p) + n).
func addptr(p ir.Node, n int64) ir.Node {
	t := p.Type()

	p = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, p)
	p.SetType(types.Types[types.TUINTPTR])

	p = ir.NewBinaryExpr(base.Pos, ir.OADD, p, ir.NewInt(n))

	p = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, p)
	p.SetType(t)

	return p
}
