// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"internal/buildcfg"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
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
	base.Assert(!nrange.DistinctVars) // Should all be rewritten before escape analysis
	if isMapClear(nrange) {
		return mapRangeClear(nrange)
	}

	nfor := ir.NewForStmt(nrange.Pos(), nil, nil, nil, nil, nrange.DistinctVars)
	nfor.SetInit(nrange.Init())
	nfor.Label = nrange.Label

	// variable name conventions:
	//	ohv1, hv1, hv2: hidden (old) val 1, 2
	//	ha, hit: hidden aggregate, iterator
	//	hn, hp: hidden len, pointer
	//	hb: hidden bool
	//	a, v1, v2: not hidden aggregate, val 1, 2

	a := nrange.X
	t := a.Type()
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

	var body []ir.Node
	var init []ir.Node
	switch k := t.Kind(); {
	default:
		base.Fatalf("walkRange")

	case types.IsInt[k]:
		hv1 := typecheck.TempAt(base.Pos, ir.CurFunc, t)
		hn := typecheck.TempAt(base.Pos, ir.CurFunc, t)

		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		init = append(init, ir.NewAssignStmt(base.Pos, hn, a))

		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn)
		nfor.Post = ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, ir.NewInt(base.Pos, 1)))

		if v1 != nil {
			body = []ir.Node{rangeAssign(nrange, hv1)}
		}

	case k == types.TARRAY, k == types.TSLICE, k == types.TPTR: // TPTR is pointer-to-array
		if nn := arrayRangeClear(nrange, v1, v2, a); nn != nil {
			base.Pos = lno
			return nn
		}

		// Element type of the iteration
		var elem *types.Type
		switch t.Kind() {
		case types.TSLICE, types.TARRAY:
			elem = t.Elem()
		case types.TPTR:
			elem = t.Elem().Elem()
		}

		// order.stmt arranged for a copy of the array/slice variable if needed.
		ha := a

		hv1 := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])
		hn := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])

		init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		init = append(init, ir.NewAssignStmt(base.Pos, hn, ir.NewUnaryExpr(base.Pos, ir.OLEN, ha)))

		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, hv1, hn)
		nfor.Post = ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, ir.NewInt(base.Pos, 1)))

		// for range ha { body }
		if v1 == nil {
			break
		}

		// for v1 := range ha { body }
		if v2 == nil {
			body = []ir.Node{rangeAssign(nrange, hv1)}
			break
		}

		// for v1, v2 := range ha { body }
		if cheapComputableIndex(elem.Size()) {
			// v1, v2 = hv1, ha[hv1]
			tmp := ir.NewIndexExpr(base.Pos, ha, hv1)
			tmp.SetBounded(true)
			body = []ir.Node{rangeAssign2(nrange, hv1, tmp)}
			break
		}

		// Slice to iterate over
		var hs ir.Node
		if t.IsSlice() {
			hs = ha
		} else {
			var arr ir.Node
			if t.IsPtr() {
				arr = ha
			} else {
				arr = typecheck.NodAddr(ha)
				arr.SetType(t.PtrTo())
				arr.SetTypecheck(1)
			}
			hs = ir.NewSliceExpr(base.Pos, ir.OSLICEARR, arr, nil, nil, nil)
			// old typechecker doesn't know OSLICEARR, so we set types explicitly
			hs.SetType(types.NewSlice(elem))
			hs.SetTypecheck(1)
		}

		// We use a "pointer" to keep track of where we are in the backing array
		// of the slice hs. This pointer starts at hs.ptr and gets incremented
		// by the element size each time through the loop.
		//
		// It's tricky, though, as on the last iteration this pointer gets
		// incremented to point past the end of the backing array. We can't
		// let the garbage collector see that final out-of-bounds pointer.
		//
		// To avoid this, we keep the "pointer" alternately in 2 variables, one
		// pointer typed and one uintptr typed. Most of the time it lives in the
		// regular pointer variable, but when it might be out of bounds (after it
		// has been incremented, but before the loop condition has been checked)
		// it lives briefly in the uintptr variable.
		//
		// hp contains the pointer version (of type *T, where T is the element type).
		// It is guaranteed to always be in range, keeps the backing store alive,
		// and is updated on stack copies. If a GC occurs when this function is
		// suspended at any safepoint, this variable ensures correct operation.
		//
		// hu contains the equivalent uintptr version. It may point past the
		// end, but doesn't keep the backing store alive and doesn't get updated
		// on a stack copy. If a GC occurs while this function is on the top of
		// the stack, then the last frame is scanned conservatively and hu will
		// act as a reference to the backing array to ensure it is not collected.
		//
		// The "pointer" we're moving across the backing array lives in one
		// or the other of hp and hu as the loop proceeds.
		//
		// hp is live during most of the body of the loop. But it isn't live
		// at the very top of the loop, when we haven't checked i<n yet, and
		// it could point off the end of the backing store.
		// hu is live only at the very top and very bottom of the loop.
		// In particular, only when it cannot possibly be live across a call.
		//
		// So we do
		//   hu = uintptr(unsafe.Pointer(hs.ptr))
		//   for i := 0; i < hs.len; i++ {
		//     hp = (*T)(unsafe.Pointer(hu))
		//     v1, v2 = i, *hp
		//     ... body of loop ...
		//     hu = uintptr(unsafe.Pointer(hp)) + elemsize
		//   }
		//
		// Between the assignments to hu and the assignment back to hp, there
		// must not be any calls.

		// Pointer to current iteration position. Start on entry to the loop
		// with the pointer in hu.
		ptr := ir.NewUnaryExpr(base.Pos, ir.OSPTR, hs)
		ptr.SetBounded(true)
		huVal := ir.NewConvExpr(base.Pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], ptr)
		huVal = ir.NewConvExpr(base.Pos, ir.OCONVNOP, types.Types[types.TUINTPTR], huVal)
		hu := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUINTPTR])
		init = append(init, ir.NewAssignStmt(base.Pos, hu, huVal))

		// Convert hu to hp at the top of the loop (after the condition has been checked).
		hpVal := ir.NewConvExpr(base.Pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], hu)
		hpVal.SetCheckPtr(true) // disable checkptr on this conversion
		hpVal = ir.NewConvExpr(base.Pos, ir.OCONVNOP, elem.PtrTo(), hpVal)
		hp := typecheck.TempAt(base.Pos, ir.CurFunc, elem.PtrTo())
		body = append(body, ir.NewAssignStmt(base.Pos, hp, hpVal))

		// Assign variables on the LHS of the range statement. Use *hp to get the element.
		e := ir.NewStarExpr(base.Pos, hp)
		e.SetBounded(true)
		a := rangeAssign2(nrange, hv1, e)
		body = append(body, a)

		// Advance pointer for next iteration of the loop.
		// This reads from hp and writes to hu.
		huVal = ir.NewConvExpr(base.Pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], hp)
		huVal = ir.NewConvExpr(base.Pos, ir.OCONVNOP, types.Types[types.TUINTPTR], huVal)
		as := ir.NewAssignStmt(base.Pos, hu, ir.NewBinaryExpr(base.Pos, ir.OADD, huVal, ir.NewInt(base.Pos, elem.Size())))
		nfor.Post = ir.NewBlockStmt(base.Pos, []ir.Node{nfor.Post, as})

	case k == types.TMAP:
		// order.stmt allocated the iterator for us.
		// we only use a once, so no copy needed.
		ha := a

		hit := nrange.Prealloc
		th := hit.Type()
		// depends on layout of iterator struct.
		// See cmd/compile/internal/reflectdata/reflect.go:MapIterType
		var keysym, elemsym *types.Sym
		var iterInit, iterNext string
		if buildcfg.Experiment.SwissMap {
			keysym = th.Field(0).Sym
			elemsym = th.Field(1).Sym // ditto
			iterInit = "mapIterStart"
			iterNext = "mapIterNext"
		} else {
			keysym = th.Field(0).Sym
			elemsym = th.Field(1).Sym // ditto
			iterInit = "mapiterinit"
			iterNext = "mapiternext"
		}

		fn := typecheck.LookupRuntime(iterInit, t.Key(), t.Elem(), th)
		init = append(init, mkcallstmt1(fn, reflectdata.RangeMapRType(base.Pos, nrange), ha, typecheck.NodAddr(hit)))
		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym), typecheck.NodNil())

		fn = typecheck.LookupRuntime(iterNext, th)
		nfor.Post = mkcallstmt1(fn, typecheck.NodAddr(hit))

		key := ir.NewStarExpr(base.Pos, typecheck.ConvNop(ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, keysym), types.NewPtr(t.Key())))
		if v1 == nil {
			body = nil
		} else if v2 == nil {
			body = []ir.Node{rangeAssign(nrange, key)}
		} else {
			elem := ir.NewStarExpr(base.Pos, typecheck.ConvNop(ir.NewSelectorExpr(base.Pos, ir.ODOT, hit, elemsym), types.NewPtr(t.Elem())))
			body = []ir.Node{rangeAssign2(nrange, key, elem)}
		}

	case k == types.TCHAN:
		// order.stmt arranged for a copy of the channel variable.
		ha := a

		hv1 := typecheck.TempAt(base.Pos, ir.CurFunc, t.Elem())
		hv1.SetTypecheck(1)
		if t.Elem().HasPointers() {
			init = append(init, ir.NewAssignStmt(base.Pos, hv1, nil))
		}
		hb := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TBOOL])

		nfor.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, hb, ir.NewBool(base.Pos, false))
		lhs := []ir.Node{hv1, hb}
		rhs := []ir.Node{ir.NewUnaryExpr(base.Pos, ir.ORECV, ha)}
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2RECV, lhs, rhs)
		a.SetTypecheck(1)
		nfor.Cond = ir.InitExpr([]ir.Node{a}, nfor.Cond)
		if v1 == nil {
			body = nil
		} else {
			body = []ir.Node{rangeAssign(nrange, hv1)}
		}
		// Zero hv1. This prevents hv1 from being the sole, inaccessible
		// reference to an otherwise GC-able value during the next channel receive.
		// See issue 15281.
		body = append(body, ir.NewAssignStmt(base.Pos, hv1, nil))

	case k == types.TSTRING:
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

		hv1 := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])
		hv1t := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])
		hv2 := typecheck.TempAt(base.Pos, ir.CurFunc, types.RuneType)

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

		// On x86, hv2 <= 127 is shorter to encode than hv2 < 128
		// Doesn't hurt other archs.
		nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLE, hv2, ir.NewInt(base.Pos, utf8.RuneSelf-1))

		// hv1++
		nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, hv1, ir.NewBinaryExpr(base.Pos, ir.OADD, hv1, ir.NewInt(base.Pos, 1)))}

		// } else {
		// hv2, hv1 = decoderune(ha, hv1)
		fn := typecheck.LookupRuntime("decoderune")
		call := mkcall1(fn, fn.Type().ResultsTuple(), &nif.Else, ha, hv1)
		a := ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{hv2, hv1}, []ir.Node{call})
		nif.Else.Append(a)

		body = append(body, nif)

		if v1 != nil {
			if v2 != nil {
				// v1, v2 = hv1t, hv2
				body = append(body, rangeAssign2(nrange, hv1t, hv2))
			} else {
				// v1 = hv1t
				body = append(body, rangeAssign(nrange, hv1t))
			}
		}
	}

	typecheck.Stmts(init)

	nfor.PtrInit().Append(init...)

	typecheck.Stmts(nfor.Cond.Init())

	nfor.Cond = typecheck.Expr(nfor.Cond)
	nfor.Cond = typecheck.DefaultLit(nfor.Cond, nil)
	nfor.Post = typecheck.Stmt(nfor.Post)
	typecheck.Stmts(body)
	nfor.Body.Append(body...)
	nfor.Body.Append(nrange.Body...)

	var n ir.Node = nfor

	n = walkStmt(n)

	base.Pos = lno
	return n
}

// rangeAssign returns "n.Key = key".
func rangeAssign(n *ir.RangeStmt, key ir.Node) ir.Node {
	key = rangeConvert(n, n.Key.Type(), key, n.KeyTypeWord, n.KeySrcRType)
	return ir.NewAssignStmt(n.Pos(), n.Key, key)
}

// rangeAssign2 returns "n.Key, n.Value = key, value".
func rangeAssign2(n *ir.RangeStmt, key, value ir.Node) ir.Node {
	// Use OAS2 to correctly handle assignments
	// of the form "v1, a[v1] = range".
	key = rangeConvert(n, n.Key.Type(), key, n.KeyTypeWord, n.KeySrcRType)
	value = rangeConvert(n, n.Value.Type(), value, n.ValueTypeWord, n.ValueSrcRType)
	return ir.NewAssignListStmt(n.Pos(), ir.OAS2, []ir.Node{n.Key, n.Value}, []ir.Node{key, value})
}

// rangeConvert returns src, converted to dst if necessary. If a
// conversion is necessary, then typeWord and srcRType are copied to
// their respective ConvExpr fields.
func rangeConvert(nrange *ir.RangeStmt, dst *types.Type, src, typeWord, srcRType ir.Node) ir.Node {
	src = typecheck.Expr(src)
	if dst.Kind() == types.TBLANK || types.Identical(dst, src.Type()) {
		return src
	}

	n := ir.NewConvExpr(nrange.Pos(), ir.OCONV, dst, src)
	n.TypeWord = typeWord
	n.SrcRType = srcRType
	return typecheck.Expr(n)
}

// isMapClear checks if n is of the form:
//
//	for k := range m {
//		delete(m, k)
//	}
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

// mapRangeClear constructs a call to runtime.mapclear for the map range idiom.
func mapRangeClear(nrange *ir.RangeStmt) ir.Node {
	m := nrange.X
	origPos := ir.SetPos(m)
	defer func() { base.Pos = origPos }()

	return mapClear(m, reflectdata.RangeMapRType(base.Pos, nrange))
}

// mapClear constructs a call to runtime.mapclear for the map m.
func mapClear(m, rtyp ir.Node) ir.Node {
	t := m.Type()

	// instantiate mapclear(typ *type, hmap map[any]any)
	fn := typecheck.LookupRuntime("mapclear", t.Key(), t.Elem())
	n := mkcallstmt1(fn, rtyp, m)
	return walkStmt(typecheck.Stmt(n))
}

// Lower n into runtime·memclr if possible, for
// fast zeroing of slices and arrays (issue 5373).
// Look for instances of
//
//	for i := range a {
//		a[i] = zero
//	}
//
// in which the evaluation of a is side-effect-free.
//
// Parameters are as in walkRange: "for v1, v2 = range a".
func arrayRangeClear(loop *ir.RangeStmt, v1, v2, a ir.Node) ir.Node {
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
	x := lhs.X
	if a.Type().IsPtr() && a.Type().Elem().IsArray() {
		if s, ok := x.(*ir.StarExpr); ok && s.Op() == ir.ODEREF {
			x = s.X
		}
	}

	if !ir.SameSafeExpr(x, a) || !ir.SameSafeExpr(lhs.Index, v1) {
		return nil
	}

	if !ir.IsZero(stmt.Y) {
		return nil
	}

	return arrayClear(stmt.Pos(), a, loop)
}

// arrayClear constructs a call to runtime.memclr for fast zeroing of slices and arrays.
func arrayClear(wbPos src.XPos, a ir.Node, nrange *ir.RangeStmt) ir.Node {
	elemsize := typecheck.RangeExprType(a.Type()).Elem().Size()
	if elemsize <= 0 {
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
	n.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(base.Pos, 0))

	// hp = &a[0]
	hp := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])

	ix := ir.NewIndexExpr(base.Pos, a, ir.NewInt(base.Pos, 0))
	ix.SetBounded(true)
	addr := typecheck.ConvNop(typecheck.NodAddr(ix), types.Types[types.TUNSAFEPTR])
	n.Body.Append(ir.NewAssignStmt(base.Pos, hp, addr))

	// hn = len(a) * sizeof(elem(a))
	hn := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUINTPTR])
	mul := typecheck.Conv(ir.NewBinaryExpr(base.Pos, ir.OMUL, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(base.Pos, elemsize)), types.Types[types.TUINTPTR])
	n.Body.Append(ir.NewAssignStmt(base.Pos, hn, mul))

	var fn ir.Node
	if a.Type().Elem().HasPointers() {
		// memclrHasPointers(hp, hn)
		ir.CurFunc.SetWBPos(wbPos)
		fn = mkcallstmt("memclrHasPointers", hp, hn)
	} else {
		// memclrNoHeapPointers(hp, hn)
		fn = mkcallstmt("memclrNoHeapPointers", hp, hn)
	}

	n.Body.Append(fn)

	// For array range clear, also set "i = len(a) - 1"
	if nrange != nil {
		idx := ir.NewAssignStmt(base.Pos, nrange.Key, ir.NewBinaryExpr(base.Pos, ir.OSUB, ir.NewUnaryExpr(base.Pos, ir.OLEN, a), ir.NewInt(base.Pos, 1)))
		n.Body.Append(idx)
	}

	n.Cond = typecheck.Expr(n.Cond)
	n.Cond = typecheck.DefaultLit(n.Cond, nil)
	typecheck.Stmts(n.Body)
	return walkStmt(n)
}
