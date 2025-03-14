// Copyright 2009 The Go Authors. All rights reserved.walk/bui
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"fmt"
	"go/constant"
	"go/token"
	"internal/abi"
	"internal/buildcfg"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// Rewrite append(src, x, y, z) so that any side effects in
// x, y, z (including runtime panics) are evaluated in
// initialization statements before the append.
// For normal code generation, stop there and leave the
// rest to ssagen.
//
// For race detector, expand append(src, a [, b]* ) to
//
//	init {
//	  s := src
//	  const argc = len(args) - 1
//	  newLen := s.len + argc
//	  if uint(newLen) <= uint(s.cap) {
//	    s = s[:newLen]
//	  } else {
//	    s = growslice(s.ptr, newLen, s.cap, argc, elemType)
//	  }
//	  s[s.len - argc] = a
//	  s[s.len - argc + 1] = b
//	  ...
//	}
//	s
func walkAppend(n *ir.CallExpr, init *ir.Nodes, dst ir.Node) ir.Node {
	if !ir.SameSafeExpr(dst, n.Args[0]) {
		n.Args[0] = safeExpr(n.Args[0], init)
		n.Args[0] = walkExpr(n.Args[0], init)
	}
	walkExprListSafe(n.Args[1:], init)

	nsrc := n.Args[0]

	// walkExprListSafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	// Using cheapExpr also makes sure that the evaluation
	// of all arguments (and especially any panics) happen
	// before we begin to modify the slice in a visible way.
	ls := n.Args[1:]
	for i, n := range ls {
		n = cheapExpr(n, init)
		if !types.Identical(n.Type(), nsrc.Type().Elem()) {
			n = typecheck.AssignConv(n, nsrc.Type().Elem(), "append")
			n = walkExpr(n, init)
		}
		ls[i] = n
	}

	argc := len(n.Args) - 1
	if argc < 1 {
		return nsrc
	}

	// General case, with no function calls left as arguments.
	// Leave for ssagen, except that instrumentation requires the old form.
	if !base.Flag.Cfg.Instrumenting || base.Flag.CompilingRuntime {
		return n
	}

	var l []ir.Node

	// s = slice to append to
	s := typecheck.TempAt(base.Pos, ir.CurFunc, nsrc.Type())
	l = append(l, ir.NewAssignStmt(base.Pos, s, nsrc))

	// num = number of things to append
	num := ir.NewInt(base.Pos, int64(argc))

	// newLen := s.len + num
	newLen := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])
	l = append(l, ir.NewAssignStmt(base.Pos, newLen, ir.NewBinaryExpr(base.Pos, ir.OADD, ir.NewUnaryExpr(base.Pos, ir.OLEN, s), num)))

	// if uint(newLen) <= uint(s.cap)
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLE, typecheck.Conv(newLen, types.Types[types.TUINT]), typecheck.Conv(ir.NewUnaryExpr(base.Pos, ir.OCAP, s), types.Types[types.TUINT]))
	nif.Likely = true

	// then { s = s[:n] }
	slice := ir.NewSliceExpr(base.Pos, ir.OSLICE, s, nil, newLen, nil)
	slice.SetBounded(true)
	nif.Body = []ir.Node{
		ir.NewAssignStmt(base.Pos, s, slice),
	}

	// else { s = growslice(s.ptr, n, s.cap, a, T) }
	nif.Else = []ir.Node{
		ir.NewAssignStmt(base.Pos, s, walkGrowslice(s, nif.PtrInit(),
			ir.NewUnaryExpr(base.Pos, ir.OSPTR, s),
			newLen,
			ir.NewUnaryExpr(base.Pos, ir.OCAP, s),
			num)),
	}

	l = append(l, nif)

	ls = n.Args[1:]
	for i, n := range ls {
		// s[s.len-argc+i] = arg
		ix := ir.NewIndexExpr(base.Pos, s, ir.NewBinaryExpr(base.Pos, ir.OSUB, newLen, ir.NewInt(base.Pos, int64(argc-i))))
		ix.SetBounded(true)
		l = append(l, ir.NewAssignStmt(base.Pos, ix, n))
	}

	typecheck.Stmts(l)
	walkStmtList(l)
	init.Append(l...)
	return s
}

// growslice(ptr *T, newLen, oldCap, num int, <type>) (ret []T)
func walkGrowslice(slice *ir.Name, init *ir.Nodes, oldPtr, newLen, oldCap, num ir.Node) *ir.CallExpr {
	elemtype := slice.Type().Elem()
	fn := typecheck.LookupRuntime("growslice", elemtype, elemtype)
	elemtypeptr := reflectdata.TypePtrAt(base.Pos, elemtype)
	return mkcall1(fn, slice.Type(), init, oldPtr, newLen, oldCap, num, elemtypeptr)
}

// walkClear walks an OCLEAR node.
func walkClear(n *ir.UnaryExpr) ir.Node {
	typ := n.X.Type()
	switch {
	case typ.IsSlice():
		if n := arrayClear(n.X.Pos(), n.X, nil); n != nil {
			return n
		}
		// If n == nil, we are clearing an array which takes zero memory, do nothing.
		return ir.NewBlockStmt(n.Pos(), nil)
	case typ.IsMap():
		return mapClear(n.X, reflectdata.TypePtrAt(n.X.Pos(), n.X.Type()))
	}
	panic("unreachable")
}

// walkClose walks an OCLOSE node.
func walkClose(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	return mkcall1(chanfn("closechan", 1, n.X.Type()), nil, init, n.X)
}

// Lower copy(a, b) to a memmove call or a runtime call.
//
//	init {
//	  n := len(a)
//	  if n > len(b) { n = len(b) }
//	  if a.ptr != b.ptr { memmove(a.ptr, b.ptr, n*sizeof(elem(a))) }
//	}
//	n;
//
// Also works if b is a string.
func walkCopy(n *ir.BinaryExpr, init *ir.Nodes, runtimecall bool) ir.Node {
	if n.X.Type().Elem().HasPointers() {
		ir.CurFunc.SetWBPos(n.Pos())
		fn := writebarrierfn("typedslicecopy", n.X.Type().Elem(), n.Y.Type().Elem())
		n.X = cheapExpr(n.X, init)
		ptrL, lenL := backingArrayPtrLen(n.X)
		n.Y = cheapExpr(n.Y, init)
		ptrR, lenR := backingArrayPtrLen(n.Y)
		return mkcall1(fn, n.Type(), init, reflectdata.CopyElemRType(base.Pos, n), ptrL, lenL, ptrR, lenR)
	}

	if runtimecall {
		// rely on runtime to instrument:
		//  copy(n.Left, n.Right)
		// n.Right can be a slice or string.

		n.X = cheapExpr(n.X, init)
		ptrL, lenL := backingArrayPtrLen(n.X)
		n.Y = cheapExpr(n.Y, init)
		ptrR, lenR := backingArrayPtrLen(n.Y)

		fn := typecheck.LookupRuntime("slicecopy", ptrL.Type().Elem(), ptrR.Type().Elem())

		return mkcall1(fn, n.Type(), init, ptrL, lenL, ptrR, lenR, ir.NewInt(base.Pos, n.X.Type().Elem().Size()))
	}

	n.X = walkExpr(n.X, init)
	n.Y = walkExpr(n.Y, init)
	nl := typecheck.TempAt(base.Pos, ir.CurFunc, n.X.Type())
	nr := typecheck.TempAt(base.Pos, ir.CurFunc, n.Y.Type())
	var l []ir.Node
	l = append(l, ir.NewAssignStmt(base.Pos, nl, n.X))
	l = append(l, ir.NewAssignStmt(base.Pos, nr, n.Y))

	nfrm := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nr)
	nto := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nl)

	nlen := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TINT])

	// n = len(to)
	l = append(l, ir.NewAssignStmt(base.Pos, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nl)))

	// if n > len(frm) { n = len(frm) }
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil)

	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OGT, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nr))
	nif.Body.Append(ir.NewAssignStmt(base.Pos, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nr)))
	l = append(l, nif)

	// if to.ptr != frm.ptr { memmove( ... ) }
	ne := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.ONE, nto, nfrm), nil, nil)
	ne.Likely = true
	l = append(l, ne)

	fn := typecheck.LookupRuntime("memmove", nl.Type().Elem(), nl.Type().Elem())
	nwid := ir.Node(typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUINTPTR]))
	setwid := ir.NewAssignStmt(base.Pos, nwid, typecheck.Conv(nlen, types.Types[types.TUINTPTR]))
	ne.Body.Append(setwid)
	nwid = ir.NewBinaryExpr(base.Pos, ir.OMUL, nwid, ir.NewInt(base.Pos, nl.Type().Elem().Size()))
	call := mkcall1(fn, nil, init, nto, nfrm, nwid)
	ne.Body.Append(call)

	typecheck.Stmts(l)
	walkStmtList(l)
	init.Append(l...)
	return nlen
}

// walkDelete walks an ODELETE node.
func walkDelete(init *ir.Nodes, n *ir.CallExpr) ir.Node {
	init.Append(ir.TakeInit(n)...)
	map_ := n.Args[0]
	key := n.Args[1]
	map_ = walkExpr(map_, init)
	key = walkExpr(key, init)

	t := map_.Type()
	fast := mapfast(t)
	key = mapKeyArg(fast, n, key, false)
	return mkcall1(mapfndel(mapdelete[fast], t), nil, init, reflectdata.DeleteMapRType(base.Pos, n), map_, key)
}

// walkLenCap walks an OLEN or OCAP node.
func walkLenCap(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	if isRuneCount(n) {
		// Replace len([]rune(string)) with runtime.countrunes(string).
		return mkcall("countrunes", n.Type(), init, typecheck.Conv(n.X.(*ir.ConvExpr).X, types.Types[types.TSTRING]))
	}
	if isByteCount(n) {
		conv := n.X.(*ir.ConvExpr)
		walkStmtList(conv.Init())
		init.Append(ir.TakeInit(conv)...)
		_, len := backingArrayPtrLen(cheapExpr(conv.X, init))
		return len
	}
	if isChanLenCap(n) {
		name := "chanlen"
		if n.Op() == ir.OCAP {
			name = "chancap"
		}
		// cannot use chanfn - closechan takes any, not chan any,
		// because it accepts both send-only and recv-only channels.
		fn := typecheck.LookupRuntime(name, n.X.Type())
		return mkcall1(fn, n.Type(), init, n.X)
	}

	n.X = walkExpr(n.X, init)

	// replace len(*[10]int) with 10.
	// delayed until now to preserve side effects.
	t := n.X.Type()
	if t.IsPtr() {
		t = t.Elem()
	}
	if t.IsArray() {
		// evaluate any side effects in n.X. See issue 72844.
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.BlankNode, n.X))

		con := ir.NewConstExpr(constant.MakeInt64(t.NumElem()), n)
		con.SetTypecheck(1)
		return con
	}
	return n
}

// walkMakeChan walks an OMAKECHAN node.
func walkMakeChan(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	// When size fits into int, use makechan instead of
	// makechan64, which is faster and shorter on 32 bit platforms.
	size := n.Len
	fnname := "makechan64"
	argtype := types.Types[types.TINT64]

	// Type checking guarantees that TIDEAL size is positive and fits in an int.
	// The case of size overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in makechan during runtime.
	if size.Type().IsKind(types.TIDEAL) || size.Type().Size() <= types.Types[types.TUINT].Size() {
		fnname = "makechan"
		argtype = types.Types[types.TINT]
	}

	return mkcall1(chanfn(fnname, 1, n.Type()), n.Type(), init, reflectdata.MakeChanRType(base.Pos, n), typecheck.Conv(size, argtype))
}

// walkMakeMap walks an OMAKEMAP node.
func walkMakeMap(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	if buildcfg.Experiment.SwissMap {
		return walkMakeSwissMap(n, init)
	}
	return walkMakeOldMap(n, init)
}

func walkMakeSwissMap(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	t := n.Type()
	mapType := reflectdata.SwissMapType()
	hint := n.Len

	// var m *Map
	var m ir.Node
	if n.Esc() == ir.EscNone {
		// Allocate hmap on stack.

		// var mv Map
		// m = &mv
		m = stackTempAddr(init, mapType)

		// Allocate one group pointed to by m.dirPtr on stack if hint
		// is not larger than SwissMapGroupSlots. In case hint is
		// larger, runtime.makemap will allocate on the heap.
		// Maximum key and elem size is 128 bytes, larger objects
		// are stored with an indirection. So max bucket size is 2048+eps.
		if !ir.IsConst(hint, constant.Int) ||
			constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(abi.SwissMapGroupSlots)) {

			// In case hint is larger than SwissMapGroupSlots
			// runtime.makemap will allocate on the heap, see
			// #20184
			//
			// if hint <= abi.SwissMapGroupSlots {
			//     var gv group
			//     g = &gv
			//     g.ctrl = abi.SwissMapCtrlEmpty
			//     m.dirPtr = g
			// }

			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLE, hint, ir.NewInt(base.Pos, abi.SwissMapGroupSlots)), nil, nil)
			nif.Likely = true

			groupType := reflectdata.SwissMapGroupType(t)

			// var gv group
			// g = &gv
			g := stackTempAddr(&nif.Body, groupType)

			// Can't use ir.NewInt because bit 63 is set, which
			// makes conversion to uint64 upset.
			empty := ir.NewBasicLit(base.Pos, types.UntypedInt, constant.MakeUint64(abi.SwissMapCtrlEmpty))

			// g.ctrl = abi.SwissMapCtrlEmpty
			csym := groupType.Field(0).Sym // g.ctrl see reflectdata/map_swiss.go
			ca := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, g, csym), empty)
			nif.Body.Append(ca)

			// m.dirPtr = g
			dsym := mapType.Field(2).Sym // m.dirPtr see reflectdata/map_swiss.go
			na := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, m, dsym), typecheck.ConvNop(g, types.Types[types.TUNSAFEPTR]))
			nif.Body.Append(na)
			appendWalkStmt(init, nif)
		}
	}

	if ir.IsConst(hint, constant.Int) && constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(abi.SwissMapGroupSlots)) {
		// Handling make(map[any]any) and
		// make(map[any]any, hint) where hint <= abi.SwissMapGroupSlots
		// specially allows for faster map initialization and
		// improves binary size by using calls with fewer arguments.
		// For hint <= abi.SwissMapGroupSlots no groups will be
		// allocated by makemap. Therefore, no groups need to be
		// allocated in this code path.
		if n.Esc() == ir.EscNone {
			// Only need to initialize m.seed since
			// m map has been allocated on the stack already.
			// m.seed = uintptr(rand())
			rand := mkcall("rand", types.Types[types.TUINT64], init)
			seedSym := mapType.Field(1).Sym // m.seed see reflectdata/map_swiss.go
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, m, seedSym), typecheck.Conv(rand, types.Types[types.TUINTPTR])))
			return typecheck.ConvNop(m, t)
		}
		// Call runtime.makemap_small to allocate a
		// map on the heap and initialize the map's seed field.
		fn := typecheck.LookupRuntime("makemap_small", t.Key(), t.Elem())
		return mkcall1(fn, n.Type(), init)
	}

	if n.Esc() != ir.EscNone {
		m = typecheck.NodNil()
	}

	// Map initialization with a variable or large hint is
	// more complicated. We therefore generate a call to
	// runtime.makemap to initialize hmap and allocate the
	// map buckets.

	// When hint fits into int, use makemap instead of
	// makemap64, which is faster and shorter on 32 bit platforms.
	fnname := "makemap64"
	argtype := types.Types[types.TINT64]

	// Type checking guarantees that TIDEAL hint is positive and fits in an int.
	// See checkmake call in TMAP case of OMAKE case in OpSwitch in typecheck1 function.
	// The case of hint overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in makemap during runtime.
	if hint.Type().IsKind(types.TIDEAL) || hint.Type().Size() <= types.Types[types.TUINT].Size() {
		fnname = "makemap"
		argtype = types.Types[types.TINT]
	}

	fn := typecheck.LookupRuntime(fnname, mapType, t.Key(), t.Elem())
	return mkcall1(fn, n.Type(), init, reflectdata.MakeMapRType(base.Pos, n), typecheck.Conv(hint, argtype), m)
}

func walkMakeOldMap(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	t := n.Type()
	hmapType := reflectdata.OldMapType()
	hint := n.Len

	// var h *hmap
	var h ir.Node
	if n.Esc() == ir.EscNone {
		// Allocate hmap on stack.

		// var hv hmap
		// h = &hv
		h = stackTempAddr(init, hmapType)

		// Allocate one bucket pointed to by hmap.buckets on stack if hint
		// is not larger than BUCKETSIZE. In case hint is larger than
		// BUCKETSIZE runtime.makemap will allocate the buckets on the heap.
		// Maximum key and elem size is 128 bytes, larger objects
		// are stored with an indirection. So max bucket size is 2048+eps.
		if !ir.IsConst(hint, constant.Int) ||
			constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(abi.OldMapBucketCount)) {

			// In case hint is larger than BUCKETSIZE runtime.makemap
			// will allocate the buckets on the heap, see #20184
			//
			// if hint <= BUCKETSIZE {
			//     var bv bmap
			//     b = &bv
			//     h.buckets = b
			// }

			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLE, hint, ir.NewInt(base.Pos, abi.OldMapBucketCount)), nil, nil)
			nif.Likely = true

			// var bv bmap
			// b = &bv
			b := stackTempAddr(&nif.Body, reflectdata.OldMapBucketType(t))

			// h.buckets = b
			bsym := hmapType.Field(5).Sym // hmap.buckets see reflect.go:hmap
			na := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, bsym), typecheck.ConvNop(b, types.Types[types.TUNSAFEPTR]))
			nif.Body.Append(na)
			appendWalkStmt(init, nif)
		}
	}

	if ir.IsConst(hint, constant.Int) && constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(abi.OldMapBucketCount)) {
		// Handling make(map[any]any) and
		// make(map[any]any, hint) where hint <= BUCKETSIZE
		// special allows for faster map initialization and
		// improves binary size by using calls with fewer arguments.
		// For hint <= BUCKETSIZE overLoadFactor(hint, 0) is false
		// and no buckets will be allocated by makemap. Therefore,
		// no buckets need to be allocated in this code path.
		if n.Esc() == ir.EscNone {
			// Only need to initialize h.hash0 since
			// hmap h has been allocated on the stack already.
			// h.hash0 = rand32()
			rand := mkcall("rand32", types.Types[types.TUINT32], init)
			hashsym := hmapType.Field(4).Sym // hmap.hash0 see reflect.go:hmap
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, hashsym), rand))
			return typecheck.ConvNop(h, t)
		}
		// Call runtime.makemap_small to allocate an
		// hmap on the heap and initialize hmap's hash0 field.
		fn := typecheck.LookupRuntime("makemap_small", t.Key(), t.Elem())
		return mkcall1(fn, n.Type(), init)
	}

	if n.Esc() != ir.EscNone {
		h = typecheck.NodNil()
	}
	// Map initialization with a variable or large hint is
	// more complicated. We therefore generate a call to
	// runtime.makemap to initialize hmap and allocate the
	// map buckets.

	// When hint fits into int, use makemap instead of
	// makemap64, which is faster and shorter on 32 bit platforms.
	fnname := "makemap64"
	argtype := types.Types[types.TINT64]

	// Type checking guarantees that TIDEAL hint is positive and fits in an int.
	// See checkmake call in TMAP case of OMAKE case in OpSwitch in typecheck1 function.
	// The case of hint overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in makemap during runtime.
	if hint.Type().IsKind(types.TIDEAL) || hint.Type().Size() <= types.Types[types.TUINT].Size() {
		fnname = "makemap"
		argtype = types.Types[types.TINT]
	}

	fn := typecheck.LookupRuntime(fnname, hmapType, t.Key(), t.Elem())
	return mkcall1(fn, n.Type(), init, reflectdata.MakeMapRType(base.Pos, n), typecheck.Conv(hint, argtype), h)
}

// walkMakeSlice walks an OMAKESLICE node.
func walkMakeSlice(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	len := n.Len
	cap := n.Cap
	len = safeExpr(len, init)
	if cap != nil {
		cap = safeExpr(cap, init)
	} else {
		cap = len
	}
	t := n.Type()
	if t.Elem().NotInHeap() {
		base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
	}

	tryStack := false
	if n.Esc() == ir.EscNone {
		if why := escape.HeapAllocReason(n); why != "" {
			base.Fatalf("%v has EscNone, but %v", n, why)
		}
		if ir.IsSmallIntConst(cap) {
			// Constant backing array - allocate it and slice it.
			cap := typecheck.IndexConst(cap)
			// Note that len might not be constant. If it isn't, check for panics.
			// cap is constrained to [0,2^31) or [0,2^63) depending on whether
			// we're in 32-bit or 64-bit systems. So it's safe to do:
			//
			// if uint64(len) > cap {
			//     if len < 0 { panicmakeslicelen() }
			//     panicmakeslicecap()
			// }
			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(len, types.Types[types.TUINT64]), ir.NewInt(base.Pos, cap)), nil, nil)
			niflen := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, len, ir.NewInt(base.Pos, 0)), nil, nil)
			niflen.Body = []ir.Node{mkcall("panicmakeslicelen", nil, init)}
			nif.Body.Append(niflen, mkcall("panicmakeslicecap", nil, init))
			init.Append(typecheck.Stmt(nif))

			// var arr [cap]E
			// s = arr[:len]
			t := types.NewArray(t.Elem(), cap) // [cap]E
			arr := typecheck.TempAt(base.Pos, ir.CurFunc, t)
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, arr, nil))    // zero temp
			s := ir.NewSliceExpr(base.Pos, ir.OSLICE, arr, nil, len, nil) // arr[:len]
			// The conv is necessary in case n.Type is named.
			return walkExpr(typecheck.Expr(typecheck.Conv(s, n.Type())), init)
		}
		// Check that this optimization is enabled in general and for this node.
		tryStack = base.Flag.N == 0 && base.VariableMakeHash.MatchPos(n.Pos(), nil)
	}

	// The final result is assigned to this variable.
	slice := typecheck.TempAt(base.Pos, ir.CurFunc, n.Type()) // []E result (possibly named)

	if tryStack {
		// K := maxStackSize/sizeof(E)
		// if cap <= K {
		//     var arr [K]E
		//     slice = arr[:len:cap]
		// } else {
		//     slice = makeslice(elemType, len, cap)
		// }
		maxStackSize := int64(base.Debug.VariableMakeThreshold)
		K := maxStackSize / t.Elem().Size() // rounds down
		if K > 0 {                          // skip if elem size is too big.
			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLE, typecheck.Conv(cap, types.Types[types.TUINT64]), ir.NewInt(base.Pos, K)), nil, nil)

			// cap is in bounds after the K check, but len might not be.
			// (Note that the slicing below would generate a panic for
			// the same bad cases, but we want makeslice panics, not
			// regular slicing panics.)
			lenCap := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(len, types.Types[types.TUINT64]), typecheck.Conv(cap, types.Types[types.TUINT64])), nil, nil)
			lenZero := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, len, ir.NewInt(base.Pos, 0)), nil, nil)
			lenZero.Body.Append(mkcall("panicmakeslicelen", nil, &lenZero.Body))
			lenCap.Body.Append(lenZero)
			lenCap.Body.Append(mkcall("panicmakeslicecap", nil, &lenCap.Body))
			nif.Body.Append(lenCap)

			t := types.NewArray(t.Elem(), K)                              // [K]E
			arr := typecheck.TempAt(base.Pos, ir.CurFunc, t)              // var arr [K]E
			nif.Body.Append(ir.NewAssignStmt(base.Pos, arr, nil))         // arr = {} (zero it)
			s := ir.NewSliceExpr(base.Pos, ir.OSLICE, arr, nil, len, cap) // arr[:len:cap]
			nif.Body.Append(ir.NewAssignStmt(base.Pos, slice, s))         // slice = arr[:len:cap]

			appendWalkStmt(init, typecheck.Stmt(nif))

			// Put makeslice call below in the else branch.
			init = &nif.Else
		}
	}

	// Set up a call to makeslice.
	// When len and cap can fit into int, use makeslice instead of
	// makeslice64, which is faster and shorter on 32 bit platforms.
	fnname := "makeslice64"
	argtype := types.Types[types.TINT64]

	// Type checking guarantees that TIDEAL len/cap are positive and fit in an int.
	// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in makeslice during runtime.
	if (len.Type().IsKind(types.TIDEAL) || len.Type().Size() <= types.Types[types.TUINT].Size()) &&
		(cap.Type().IsKind(types.TIDEAL) || cap.Type().Size() <= types.Types[types.TUINT].Size()) {
		fnname = "makeslice"
		argtype = types.Types[types.TINT]
	}
	fn := typecheck.LookupRuntime(fnname)
	ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, reflectdata.MakeSliceElemRType(base.Pos, n), typecheck.Conv(len, argtype), typecheck.Conv(cap, argtype))
	ptr.MarkNonNil()
	len = typecheck.Conv(len, types.Types[types.TINT])
	cap = typecheck.Conv(cap, types.Types[types.TINT])
	s := ir.NewSliceHeaderExpr(base.Pos, t, ptr, len, cap)
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, slice, s))

	return slice
}

// walkMakeSliceCopy walks an OMAKESLICECOPY node.
func walkMakeSliceCopy(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	if n.Esc() == ir.EscNone {
		base.Fatalf("OMAKESLICECOPY with EscNone: %v", n)
	}

	t := n.Type()
	if t.Elem().NotInHeap() {
		base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
	}

	length := typecheck.Conv(n.Len, types.Types[types.TINT])
	copylen := ir.NewUnaryExpr(base.Pos, ir.OLEN, n.Cap)
	copyptr := ir.NewUnaryExpr(base.Pos, ir.OSPTR, n.Cap)

	if !t.Elem().HasPointers() && n.Bounded() {
		// When len(to)==len(from) and elements have no pointers:
		// replace make+copy with runtime.mallocgc+runtime.memmove.

		// We do not check for overflow of len(to)*elem.Width here
		// since len(from) is an existing checked slice capacity
		// with same elem.Width for the from slice.
		size := ir.NewBinaryExpr(base.Pos, ir.OMUL, typecheck.Conv(length, types.Types[types.TUINTPTR]), typecheck.Conv(ir.NewInt(base.Pos, t.Elem().Size()), types.Types[types.TUINTPTR]))

		// instantiate mallocgc(size uintptr, typ *byte, needszero bool) unsafe.Pointer
		fn := typecheck.LookupRuntime("mallocgc")
		ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, size, typecheck.NodNil(), ir.NewBool(base.Pos, false))
		ptr.MarkNonNil()
		sh := ir.NewSliceHeaderExpr(base.Pos, t, ptr, length, length)

		s := typecheck.TempAt(base.Pos, ir.CurFunc, t)
		r := typecheck.Stmt(ir.NewAssignStmt(base.Pos, s, sh))
		r = walkExpr(r, init)
		init.Append(r)

		// instantiate memmove(to *any, frm *any, size uintptr)
		fn = typecheck.LookupRuntime("memmove", t.Elem(), t.Elem())
		ncopy := mkcall1(fn, nil, init, ir.NewUnaryExpr(base.Pos, ir.OSPTR, s), copyptr, size)
		init.Append(walkExpr(typecheck.Stmt(ncopy), init))

		return s
	}
	// Replace make+copy with runtime.makeslicecopy.
	// instantiate makeslicecopy(typ *byte, tolen int, fromlen int, from unsafe.Pointer) unsafe.Pointer
	fn := typecheck.LookupRuntime("makeslicecopy")
	ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, reflectdata.MakeSliceElemRType(base.Pos, n), length, copylen, typecheck.Conv(copyptr, types.Types[types.TUNSAFEPTR]))
	ptr.MarkNonNil()
	sh := ir.NewSliceHeaderExpr(base.Pos, t, ptr, length, length)
	return walkExpr(typecheck.Expr(sh), init)
}

// walkNew walks an ONEW node.
func walkNew(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	t := n.Type().Elem()
	if t.NotInHeap() {
		base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", n.Type().Elem())
	}
	if n.Esc() == ir.EscNone {
		if t.Size() > ir.MaxImplicitStackVarSize {
			base.Fatalf("large ONEW with EscNone: %v", n)
		}
		return stackTempAddr(init, t)
	}
	types.CalcSize(t)
	n.MarkNonNil()
	return n
}

func walkMinMax(n *ir.CallExpr, init *ir.Nodes) ir.Node {
	init.Append(ir.TakeInit(n)...)
	walkExprList(n.Args, init)
	return n
}

// generate code for print.
func walkPrint(nn *ir.CallExpr, init *ir.Nodes) ir.Node {
	// Hoist all the argument evaluation up before the lock.
	walkExprListCheap(nn.Args, init)

	// For println, add " " between elements and "\n" at the end.
	if nn.Op() == ir.OPRINTLN {
		s := nn.Args
		t := make([]ir.Node, 0, len(s)*2)
		for i, n := range s {
			if i != 0 {
				t = append(t, ir.NewString(base.Pos, " "))
			}
			t = append(t, n)
		}
		t = append(t, ir.NewString(base.Pos, "\n"))
		nn.Args = t
	}

	// Collapse runs of constant strings.
	s := nn.Args
	t := make([]ir.Node, 0, len(s))
	for i := 0; i < len(s); {
		var strs []string
		for i < len(s) && ir.IsConst(s[i], constant.String) {
			strs = append(strs, ir.StringVal(s[i]))
			i++
		}
		if len(strs) > 0 {
			t = append(t, ir.NewString(base.Pos, strings.Join(strs, "")))
		}
		if i < len(s) {
			t = append(t, s[i])
			i++
		}
	}
	nn.Args = t

	calls := []ir.Node{mkcall("printlock", nil, init)}
	for i, n := range nn.Args {
		if n.Op() == ir.OLITERAL {
			if n.Type() == types.UntypedRune {
				n = typecheck.DefaultLit(n, types.RuneType)
			}

			switch n.Val().Kind() {
			case constant.Int:
				n = typecheck.DefaultLit(n, types.Types[types.TINT64])

			case constant.Float:
				n = typecheck.DefaultLit(n, types.Types[types.TFLOAT64])
			}
		}

		if n.Op() != ir.OLITERAL && n.Type() != nil && n.Type().Kind() == types.TIDEAL {
			n = typecheck.DefaultLit(n, types.Types[types.TINT64])
		}
		n = typecheck.DefaultLit(n, nil)
		nn.Args[i] = n
		if n.Type() == nil || n.Type().Kind() == types.TFORW {
			continue
		}

		var on *ir.Name
		switch n.Type().Kind() {
		case types.TINTER:
			if n.Type().IsEmptyInterface() {
				on = typecheck.LookupRuntime("printeface", n.Type())
			} else {
				on = typecheck.LookupRuntime("printiface", n.Type())
			}
		case types.TPTR:
			if n.Type().Elem().NotInHeap() {
				on = typecheck.LookupRuntime("printuintptr")
				n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
				n.SetType(types.Types[types.TUNSAFEPTR])
				n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
				n.SetType(types.Types[types.TUINTPTR])
				break
			}
			fallthrough
		case types.TCHAN, types.TMAP, types.TFUNC, types.TUNSAFEPTR:
			on = typecheck.LookupRuntime("printpointer", n.Type())
		case types.TSLICE:
			on = typecheck.LookupRuntime("printslice", n.Type())
		case types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR:
			if types.RuntimeSymName(n.Type().Sym()) == "hex" {
				on = typecheck.LookupRuntime("printhex")
			} else {
				on = typecheck.LookupRuntime("printuint")
			}
		case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64:
			on = typecheck.LookupRuntime("printint")
		case types.TFLOAT32, types.TFLOAT64:
			on = typecheck.LookupRuntime("printfloat")
		case types.TCOMPLEX64, types.TCOMPLEX128:
			on = typecheck.LookupRuntime("printcomplex")
		case types.TBOOL:
			on = typecheck.LookupRuntime("printbool")
		case types.TSTRING:
			cs := ""
			if ir.IsConst(n, constant.String) {
				cs = ir.StringVal(n)
			}
			switch cs {
			case " ":
				on = typecheck.LookupRuntime("printsp")
			case "\n":
				on = typecheck.LookupRuntime("printnl")
			default:
				on = typecheck.LookupRuntime("printstring")
			}
		default:
			badtype(ir.OPRINT, n.Type(), nil)
			continue
		}

		r := ir.NewCallExpr(base.Pos, ir.OCALL, on, nil)
		if params := on.Type().Params(); len(params) > 0 {
			t := params[0].Type
			n = typecheck.Conv(n, t)
			r.Args.Append(n)
		}
		calls = append(calls, r)
	}

	calls = append(calls, mkcall("printunlock", nil, init))

	typecheck.Stmts(calls)
	walkExprList(calls, init)

	r := ir.NewBlockStmt(base.Pos, nil)
	r.List = calls
	return walkStmt(typecheck.Stmt(r))
}

// walkRecoverFP walks an ORECOVERFP node.
func walkRecoverFP(nn *ir.CallExpr, init *ir.Nodes) ir.Node {
	return mkcall("gorecover", nn.Type(), init, walkExpr(nn.Args[0], init))
}

// walkUnsafeData walks an OUNSAFESLICEDATA or OUNSAFESTRINGDATA expression.
func walkUnsafeData(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	slice := walkExpr(n.X, init)
	res := typecheck.Expr(ir.NewUnaryExpr(n.Pos(), ir.OSPTR, slice))
	res.SetType(n.Type())
	return walkExpr(res, init)
}

func walkUnsafeSlice(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	ptr := safeExpr(n.X, init)
	len := safeExpr(n.Y, init)
	sliceType := n.Type()

	lenType := types.Types[types.TINT64]
	unsafePtr := typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR])

	// If checkptr enabled, call runtime.unsafeslicecheckptr to check ptr and len.
	// for simplicity, unsafeslicecheckptr always uses int64.
	// Type checking guarantees that TIDEAL len/cap are positive and fit in an int.
	// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in unsafeslice during runtime.
	if ir.ShouldCheckPtr(ir.CurFunc, 1) {
		fnname := "unsafeslicecheckptr"
		fn := typecheck.LookupRuntime(fnname)
		init.Append(mkcall1(fn, nil, init, reflectdata.UnsafeSliceElemRType(base.Pos, n), unsafePtr, typecheck.Conv(len, lenType)))
	} else {
		// Otherwise, open code unsafe.Slice to prevent runtime call overhead.
		// Keep this code in sync with runtime.unsafeslice{,64}
		if len.Type().IsKind(types.TIDEAL) || len.Type().Size() <= types.Types[types.TUINT].Size() {
			lenType = types.Types[types.TINT]
		} else {
			// len64 := int64(len)
			// if int64(int(len64)) != len64 {
			//     panicunsafeslicelen()
			// }
			len64 := typecheck.Conv(len, lenType)
			nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
			nif.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, typecheck.Conv(typecheck.Conv(len64, types.Types[types.TINT]), lenType), len64)
			nif.Body.Append(mkcall("panicunsafeslicelen", nil, &nif.Body))
			appendWalkStmt(init, nif)
		}

		// if len < 0 { panicunsafeslicelen() }
		nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, typecheck.Conv(len, lenType), ir.NewInt(base.Pos, 0))
		nif.Body.Append(mkcall("panicunsafeslicelen", nil, &nif.Body))
		appendWalkStmt(init, nif)

		if sliceType.Elem().Size() == 0 {
			// if ptr == nil && len > 0  {
			//      panicunsafesliceptrnil()
			// }
			nifPtr := ir.NewIfStmt(base.Pos, nil, nil, nil)
			isNil := ir.NewBinaryExpr(base.Pos, ir.OEQ, unsafePtr, typecheck.NodNil())
			gtZero := ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(len, lenType), ir.NewInt(base.Pos, 0))
			nifPtr.Cond =
				ir.NewLogicalExpr(base.Pos, ir.OANDAND, isNil, gtZero)
			nifPtr.Body.Append(mkcall("panicunsafeslicenilptr", nil, &nifPtr.Body))
			appendWalkStmt(init, nifPtr)

			h := ir.NewSliceHeaderExpr(n.Pos(), sliceType,
				typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR]),
				typecheck.Conv(len, types.Types[types.TINT]),
				typecheck.Conv(len, types.Types[types.TINT]))
			return walkExpr(typecheck.Expr(h), init)
		}

		// mem, overflow := math.mulUintptr(et.size, len)
		mem := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUINTPTR])
		overflow := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TBOOL])

		decl := types.NewSignature(nil,
			[]*types.Field{
				types.NewField(base.Pos, nil, types.Types[types.TUINTPTR]),
				types.NewField(base.Pos, nil, types.Types[types.TUINTPTR]),
			},
			[]*types.Field{
				types.NewField(base.Pos, nil, types.Types[types.TUINTPTR]),
				types.NewField(base.Pos, nil, types.Types[types.TBOOL]),
			})

		fn := ir.NewFunc(n.Pos(), n.Pos(), math_MulUintptr, decl)

		call := mkcall1(fn.Nname, fn.Type().ResultsTuple(), init, ir.NewInt(base.Pos, sliceType.Elem().Size()), typecheck.Conv(typecheck.Conv(len, lenType), types.Types[types.TUINTPTR]))
		appendWalkStmt(init, ir.NewAssignListStmt(base.Pos, ir.OAS2, []ir.Node{mem, overflow}, []ir.Node{call}))

		// if overflow || mem > -uintptr(ptr) {
		//     if ptr == nil {
		//         panicunsafesliceptrnil()
		//     }
		//     panicunsafeslicelen()
		// }
		nif = ir.NewIfStmt(base.Pos, nil, nil, nil)
		memCond := ir.NewBinaryExpr(base.Pos, ir.OGT, mem, ir.NewUnaryExpr(base.Pos, ir.ONEG, typecheck.Conv(unsafePtr, types.Types[types.TUINTPTR])))
		nif.Cond = ir.NewLogicalExpr(base.Pos, ir.OOROR, overflow, memCond)
		nifPtr := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nifPtr.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, unsafePtr, typecheck.NodNil())
		nifPtr.Body.Append(mkcall("panicunsafeslicenilptr", nil, &nifPtr.Body))
		nif.Body.Append(nifPtr, mkcall("panicunsafeslicelen", nil, &nif.Body))
		appendWalkStmt(init, nif)
	}

	h := ir.NewSliceHeaderExpr(n.Pos(), sliceType,
		typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR]),
		typecheck.Conv(len, types.Types[types.TINT]),
		typecheck.Conv(len, types.Types[types.TINT]))
	return walkExpr(typecheck.Expr(h), init)
}

var math_MulUintptr = &types.Sym{Pkg: types.NewPkg("internal/runtime/math", "math"), Name: "MulUintptr"}

func walkUnsafeString(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	ptr := safeExpr(n.X, init)
	len := safeExpr(n.Y, init)

	lenType := types.Types[types.TINT64]
	unsafePtr := typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR])

	// If checkptr enabled, call runtime.unsafestringcheckptr to check ptr and len.
	// for simplicity, unsafestringcheckptr always uses int64.
	// Type checking guarantees that TIDEAL len are positive and fit in an int.
	if ir.ShouldCheckPtr(ir.CurFunc, 1) {
		fnname := "unsafestringcheckptr"
		fn := typecheck.LookupRuntime(fnname)
		init.Append(mkcall1(fn, nil, init, unsafePtr, typecheck.Conv(len, lenType)))
	} else {
		// Otherwise, open code unsafe.String to prevent runtime call overhead.
		// Keep this code in sync with runtime.unsafestring{,64}
		if len.Type().IsKind(types.TIDEAL) || len.Type().Size() <= types.Types[types.TUINT].Size() {
			lenType = types.Types[types.TINT]
		} else {
			// len64 := int64(len)
			// if int64(int(len64)) != len64 {
			//     panicunsafestringlen()
			// }
			len64 := typecheck.Conv(len, lenType)
			nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
			nif.Cond = ir.NewBinaryExpr(base.Pos, ir.ONE, typecheck.Conv(typecheck.Conv(len64, types.Types[types.TINT]), lenType), len64)
			nif.Body.Append(mkcall("panicunsafestringlen", nil, &nif.Body))
			appendWalkStmt(init, nif)
		}

		// if len < 0 { panicunsafestringlen() }
		nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, typecheck.Conv(len, lenType), ir.NewInt(base.Pos, 0))
		nif.Body.Append(mkcall("panicunsafestringlen", nil, &nif.Body))
		appendWalkStmt(init, nif)

		// if uintpr(len) > -uintptr(ptr) {
		//    if ptr == nil {
		//       panicunsafestringnilptr()
		//    }
		//    panicunsafeslicelen()
		// }
		nifLen := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nifLen.Cond = ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(len, types.Types[types.TUINTPTR]), ir.NewUnaryExpr(base.Pos, ir.ONEG, typecheck.Conv(unsafePtr, types.Types[types.TUINTPTR])))
		nifPtr := ir.NewIfStmt(base.Pos, nil, nil, nil)
		nifPtr.Cond = ir.NewBinaryExpr(base.Pos, ir.OEQ, unsafePtr, typecheck.NodNil())
		nifPtr.Body.Append(mkcall("panicunsafestringnilptr", nil, &nifPtr.Body))
		nifLen.Body.Append(nifPtr, mkcall("panicunsafestringlen", nil, &nifLen.Body))
		appendWalkStmt(init, nifLen)
	}
	h := ir.NewStringHeaderExpr(n.Pos(),
		typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR]),
		typecheck.Conv(len, types.Types[types.TINT]),
	)
	return walkExpr(typecheck.Expr(h), init)
}

func badtype(op ir.Op, tl, tr *types.Type) {
	var s string
	if tl != nil {
		s += fmt.Sprintf("\n\t%v", tl)
	}
	if tr != nil {
		s += fmt.Sprintf("\n\t%v", tr)
	}

	// common mistake: *struct and *interface.
	if tl != nil && tr != nil && tl.IsPtr() && tr.IsPtr() {
		if tl.Elem().IsStruct() && tr.Elem().IsInterface() {
			s += "\n\t(*struct vs *interface)"
		} else if tl.Elem().IsInterface() && tr.Elem().IsStruct() {
			s += "\n\t(*interface vs *struct)"
		}
	}

	base.Errorf("illegal types for operand: %v%s", op, s)
}

func writebarrierfn(name string, l *types.Type, r *types.Type) ir.Node {
	return typecheck.LookupRuntime(name, l, r)
}

// isRuneCount reports whether n is of the form len([]rune(string)).
// These are optimized into a call to runtime.countrunes.
func isRuneCount(n ir.Node) bool {
	return base.Flag.N == 0 && !base.Flag.Cfg.Instrumenting && n.Op() == ir.OLEN && n.(*ir.UnaryExpr).X.Op() == ir.OSTR2RUNES
}

// isByteCount reports whether n is of the form len(string([]byte)).
func isByteCount(n ir.Node) bool {
	return base.Flag.N == 0 && !base.Flag.Cfg.Instrumenting && n.Op() == ir.OLEN &&
		(n.(*ir.UnaryExpr).X.Op() == ir.OBYTES2STR || n.(*ir.UnaryExpr).X.Op() == ir.OBYTES2STRTMP)
}

// isChanLenCap reports whether n is of the form len(c) or cap(c) for a channel c.
// Note that this does not check for -n or instrumenting because this
// is a correctness rewrite, not an optimization.
func isChanLenCap(n ir.Node) bool {
	return (n.Op() == ir.OLEN || n.Op() == ir.OCAP) && n.(*ir.UnaryExpr).X.Type().IsChan()
}
