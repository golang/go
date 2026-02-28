// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"fmt"
	"go/constant"
	"go/token"
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
// rest to cgen_append.
//
// For race detector, expand append(src, a [, b]* ) to
//
//	  init {
//	    s := src
//	    const argc = len(args) - 1
//	    if cap(s) - len(s) < argc {
//		    s = growslice(s, len(s)+argc)
//	    }
//	    n := len(s)
//	    s = s[:n+argc]
//	    s[n] = a
//	    s[n+1] = b
//	    ...
//	  }
//	  s
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
	// Leave for gen, except that instrumentation requires old form.
	if !base.Flag.Cfg.Instrumenting || base.Flag.CompilingRuntime {
		return n
	}

	var l []ir.Node

	ns := typecheck.Temp(nsrc.Type())
	l = append(l, ir.NewAssignStmt(base.Pos, ns, nsrc)) // s = src

	na := ir.NewInt(int64(argc))                 // const argc
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil) // if cap(s) - len(s) < argc
	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, ir.NewBinaryExpr(base.Pos, ir.OSUB, ir.NewUnaryExpr(base.Pos, ir.OCAP, ns), ir.NewUnaryExpr(base.Pos, ir.OLEN, ns)), na)

	fn := typecheck.LookupRuntime("growslice") //   growslice(<type>, old []T, mincap int) (ret []T)
	fn = typecheck.SubstArgTypes(fn, ns.Type().Elem(), ns.Type().Elem())

	nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, ns, mkcall1(fn, ns.Type(), nif.PtrInit(), reflectdata.TypePtr(ns.Type().Elem()), ns,
		ir.NewBinaryExpr(base.Pos, ir.OADD, ir.NewUnaryExpr(base.Pos, ir.OLEN, ns), na)))}

	l = append(l, nif)

	nn := typecheck.Temp(types.Types[types.TINT])
	l = append(l, ir.NewAssignStmt(base.Pos, nn, ir.NewUnaryExpr(base.Pos, ir.OLEN, ns))) // n = len(s)

	slice := ir.NewSliceExpr(base.Pos, ir.OSLICE, ns, nil, ir.NewBinaryExpr(base.Pos, ir.OADD, nn, na), nil) // ...s[:n+argc]
	slice.SetBounded(true)
	l = append(l, ir.NewAssignStmt(base.Pos, ns, slice)) // s = s[:n+argc]

	ls = n.Args[1:]
	for i, n := range ls {
		ix := ir.NewIndexExpr(base.Pos, ns, nn) // s[n] ...
		ix.SetBounded(true)
		l = append(l, ir.NewAssignStmt(base.Pos, ix, n)) // s[n] = arg
		if i+1 < len(ls) {
			l = append(l, ir.NewAssignStmt(base.Pos, nn, ir.NewBinaryExpr(base.Pos, ir.OADD, nn, ir.NewInt(1)))) // n = n + 1
		}
	}

	typecheck.Stmts(l)
	walkStmtList(l)
	init.Append(l...)
	return ns
}

// walkClose walks an OCLOSE node.
func walkClose(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	// cannot use chanfn - closechan takes any, not chan any
	fn := typecheck.LookupRuntime("closechan")
	fn = typecheck.SubstArgTypes(fn, n.X.Type())
	return mkcall1(fn, nil, init, n.X)
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
		return mkcall1(fn, n.Type(), init, reflectdata.TypePtr(n.X.Type().Elem()), ptrL, lenL, ptrR, lenR)
	}

	if runtimecall {
		// rely on runtime to instrument:
		//  copy(n.Left, n.Right)
		// n.Right can be a slice or string.

		n.X = cheapExpr(n.X, init)
		ptrL, lenL := backingArrayPtrLen(n.X)
		n.Y = cheapExpr(n.Y, init)
		ptrR, lenR := backingArrayPtrLen(n.Y)

		fn := typecheck.LookupRuntime("slicecopy")
		fn = typecheck.SubstArgTypes(fn, ptrL.Type().Elem(), ptrR.Type().Elem())

		return mkcall1(fn, n.Type(), init, ptrL, lenL, ptrR, lenR, ir.NewInt(n.X.Type().Elem().Size()))
	}

	n.X = walkExpr(n.X, init)
	n.Y = walkExpr(n.Y, init)
	nl := typecheck.Temp(n.X.Type())
	nr := typecheck.Temp(n.Y.Type())
	var l []ir.Node
	l = append(l, ir.NewAssignStmt(base.Pos, nl, n.X))
	l = append(l, ir.NewAssignStmt(base.Pos, nr, n.Y))

	nfrm := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nr)
	nto := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nl)

	nlen := typecheck.Temp(types.Types[types.TINT])

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

	fn := typecheck.LookupRuntime("memmove")
	fn = typecheck.SubstArgTypes(fn, nl.Type().Elem(), nl.Type().Elem())
	nwid := ir.Node(typecheck.Temp(types.Types[types.TUINTPTR]))
	setwid := ir.NewAssignStmt(base.Pos, nwid, typecheck.Conv(nlen, types.Types[types.TUINTPTR]))
	ne.Body.Append(setwid)
	nwid = ir.NewBinaryExpr(base.Pos, ir.OMUL, nwid, ir.NewInt(nl.Type().Elem().Size()))
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
	return mkcall1(mapfndel(mapdelete[fast], t), nil, init, reflectdata.TypePtr(t), map_, key)
}

// walkLenCap walks an OLEN or OCAP node.
func walkLenCap(n *ir.UnaryExpr, init *ir.Nodes) ir.Node {
	if isRuneCount(n) {
		// Replace len([]rune(string)) with runtime.countrunes(string).
		return mkcall("countrunes", n.Type(), init, typecheck.Conv(n.X.(*ir.ConvExpr).X, types.Types[types.TSTRING]))
	}

	n.X = walkExpr(n.X, init)

	// replace len(*[10]int) with 10.
	// delayed until now to preserve side effects.
	t := n.X.Type()

	if t.IsPtr() {
		t = t.Elem()
	}
	if t.IsArray() {
		safeExpr(n.X, init)
		con := typecheck.OrigInt(n, t.NumElem())
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

	return mkcall1(chanfn(fnname, 1, n.Type()), n.Type(), init, reflectdata.TypePtr(n.Type()), typecheck.Conv(size, argtype))
}

// walkMakeMap walks an OMAKEMAP node.
func walkMakeMap(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	t := n.Type()
	hmapType := reflectdata.MapType(t)
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
			constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(reflectdata.BUCKETSIZE)) {

			// In case hint is larger than BUCKETSIZE runtime.makemap
			// will allocate the buckets on the heap, see #20184
			//
			// if hint <= BUCKETSIZE {
			//     var bv bmap
			//     b = &bv
			//     h.buckets = b
			// }

			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLE, hint, ir.NewInt(reflectdata.BUCKETSIZE)), nil, nil)
			nif.Likely = true

			// var bv bmap
			// b = &bv
			b := stackTempAddr(&nif.Body, reflectdata.MapBucketType(t))

			// h.buckets = b
			bsym := hmapType.Field(5).Sym // hmap.buckets see reflect.go:hmap
			na := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, bsym), b)
			nif.Body.Append(na)
			appendWalkStmt(init, nif)
		}
	}

	if ir.IsConst(hint, constant.Int) && constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(reflectdata.BUCKETSIZE)) {
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
			// h.hash0 = fastrand()
			rand := mkcall("fastrand", types.Types[types.TUINT32], init)
			hashsym := hmapType.Field(4).Sym // hmap.hash0 see reflect.go:hmap
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, hashsym), rand))
			return typecheck.ConvNop(h, t)
		}
		// Call runtime.makehmap to allocate an
		// hmap on the heap and initialize hmap's hash0 field.
		fn := typecheck.LookupRuntime("makemap_small")
		fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem())
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

	fn := typecheck.LookupRuntime(fnname)
	fn = typecheck.SubstArgTypes(fn, hmapType, t.Key(), t.Elem())
	return mkcall1(fn, n.Type(), init, reflectdata.TypePtr(n.Type()), typecheck.Conv(hint, argtype), h)
}

// walkMakeSlice walks an OMAKESLICE node.
func walkMakeSlice(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	l := n.Len
	r := n.Cap
	if r == nil {
		r = safeExpr(l, init)
		l = r
	}
	t := n.Type()
	if t.Elem().NotInHeap() {
		base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
	}
	if n.Esc() == ir.EscNone {
		if why := escape.HeapAllocReason(n); why != "" {
			base.Fatalf("%v has EscNone, but %v", n, why)
		}
		// var arr [r]T
		// n = arr[:l]
		i := typecheck.IndexConst(r)
		if i < 0 {
			base.Fatalf("walkExpr: invalid index %v", r)
		}

		// cap is constrained to [0,2^31) or [0,2^63) depending on whether
		// we're in 32-bit or 64-bit systems. So it's safe to do:
		//
		// if uint64(len) > cap {
		//     if len < 0 { panicmakeslicelen() }
		//     panicmakeslicecap()
		// }
		nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(l, types.Types[types.TUINT64]), ir.NewInt(i)), nil, nil)
		niflen := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, l, ir.NewInt(0)), nil, nil)
		niflen.Body = []ir.Node{mkcall("panicmakeslicelen", nil, init)}
		nif.Body.Append(niflen, mkcall("panicmakeslicecap", nil, init))
		init.Append(typecheck.Stmt(nif))

		t = types.NewArray(t.Elem(), i) // [r]T
		var_ := typecheck.Temp(t)
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, nil))  // zero temp
		r := ir.NewSliceExpr(base.Pos, ir.OSLICE, var_, nil, l, nil) // arr[:l]
		// The conv is necessary in case n.Type is named.
		return walkExpr(typecheck.Expr(typecheck.Conv(r, n.Type())), init)
	}

	// n escapes; set up a call to makeslice.
	// When len and cap can fit into int, use makeslice instead of
	// makeslice64, which is faster and shorter on 32 bit platforms.

	len, cap := l, r

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
	ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, reflectdata.TypePtr(t.Elem()), typecheck.Conv(len, argtype), typecheck.Conv(cap, argtype))
	ptr.MarkNonNil()
	len = typecheck.Conv(len, types.Types[types.TINT])
	cap = typecheck.Conv(cap, types.Types[types.TINT])
	sh := ir.NewSliceHeaderExpr(base.Pos, t, ptr, len, cap)
	return walkExpr(typecheck.Expr(sh), init)
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
		size := ir.NewBinaryExpr(base.Pos, ir.OMUL, typecheck.Conv(length, types.Types[types.TUINTPTR]), typecheck.Conv(ir.NewInt(t.Elem().Size()), types.Types[types.TUINTPTR]))

		// instantiate mallocgc(size uintptr, typ *byte, needszero bool) unsafe.Pointer
		fn := typecheck.LookupRuntime("mallocgc")
		ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, size, typecheck.NodNil(), ir.NewBool(false))
		ptr.MarkNonNil()
		sh := ir.NewSliceHeaderExpr(base.Pos, t, ptr, length, length)

		s := typecheck.Temp(t)
		r := typecheck.Stmt(ir.NewAssignStmt(base.Pos, s, sh))
		r = walkExpr(r, init)
		init.Append(r)

		// instantiate memmove(to *any, frm *any, size uintptr)
		fn = typecheck.LookupRuntime("memmove")
		fn = typecheck.SubstArgTypes(fn, t.Elem(), t.Elem())
		ncopy := mkcall1(fn, nil, init, ir.NewUnaryExpr(base.Pos, ir.OSPTR, s), copyptr, size)
		init.Append(walkExpr(typecheck.Stmt(ncopy), init))

		return s
	}
	// Replace make+copy with runtime.makeslicecopy.
	// instantiate makeslicecopy(typ *byte, tolen int, fromlen int, from unsafe.Pointer) unsafe.Pointer
	fn := typecheck.LookupRuntime("makeslicecopy")
	ptr := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, reflectdata.TypePtr(t.Elem()), length, copylen, typecheck.Conv(copyptr, types.Types[types.TUNSAFEPTR]))
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

// generate code for print
func walkPrint(nn *ir.CallExpr, init *ir.Nodes) ir.Node {
	// Hoist all the argument evaluation up before the lock.
	walkExprListCheap(nn.Args, init)

	// For println, add " " between elements and "\n" at the end.
	if nn.Op() == ir.OPRINTN {
		s := nn.Args
		t := make([]ir.Node, 0, len(s)*2)
		for i, n := range s {
			if i != 0 {
				t = append(t, ir.NewString(" "))
			}
			t = append(t, n)
		}
		t = append(t, ir.NewString("\n"))
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
			t = append(t, ir.NewString(strings.Join(strs, "")))
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
				on = typecheck.LookupRuntime("printeface")
			} else {
				on = typecheck.LookupRuntime("printiface")
			}
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
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
			on = typecheck.LookupRuntime("printpointer")
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
		case types.TSLICE:
			on = typecheck.LookupRuntime("printslice")
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
		case types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR:
			if types.IsRuntimePkg(n.Type().Sym().Pkg) && n.Type().Sym().Name == "hex" {
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
		if params := on.Type().Params().FieldSlice(); len(params) > 0 {
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

// walkRecover walks an ORECOVERFP node.
func walkRecoverFP(nn *ir.CallExpr, init *ir.Nodes) ir.Node {
	return mkcall("gorecover", nn.Type(), init, walkExpr(nn.Args[0], init))
}

func walkUnsafeSlice(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	ptr := safeExpr(n.X, init)
	len := safeExpr(n.Y, init)

	fnname := "unsafeslice64"
	lenType := types.Types[types.TINT64]

	// Type checking guarantees that TIDEAL len/cap are positive and fit in an int.
	// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
	// will be handled by the negative range checks in unsafeslice during runtime.
	if ir.ShouldCheckPtr(ir.CurFunc, 1) {
		fnname = "unsafeslicecheckptr"
		// for simplicity, unsafeslicecheckptr always uses int64
	} else if len.Type().IsKind(types.TIDEAL) || len.Type().Size() <= types.Types[types.TUINT].Size() {
		fnname = "unsafeslice"
		lenType = types.Types[types.TINT]
	}

	t := n.Type()

	// Call runtime.unsafeslice{,64,checkptr} to check ptr and len.
	fn := typecheck.LookupRuntime(fnname)
	init.Append(mkcall1(fn, nil, init, reflectdata.TypePtr(t.Elem()), typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR]), typecheck.Conv(len, lenType)))

	h := ir.NewSliceHeaderExpr(n.Pos(), t,
		typecheck.Conv(ptr, types.Types[types.TUNSAFEPTR]),
		typecheck.Conv(len, types.Types[types.TINT]),
		typecheck.Conv(len, types.Types[types.TINT]))
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
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, l, r)
	return fn
}

// isRuneCount reports whether n is of the form len([]rune(string)).
// These are optimized into a call to runtime.countrunes.
func isRuneCount(n ir.Node) bool {
	return base.Flag.N == 0 && !base.Flag.Cfg.Instrumenting && n.Op() == ir.OLEN && n.(*ir.UnaryExpr).X.Op() == ir.OSTR2RUNES
}
