// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"internal/buildcfg"
)

var logosArenaRetTok = make(map[*ir.Func]*ir.Name)

func installLogosArenaFrame(fn *ir.Func) {
	if !escape.LogosArenaPlaceable(fn) {
		return
	}
	isCandidate := escape.LogosArenaCandidate(fn)
	needsFrame := escape.LogosArenaNeedsFrame(fn)
	if !needsFrame && !isCandidate {
		return
	}

	pos := fn.Pos()
	var takeStmt ir.Node
	var tok *ir.Name
	if isCandidate {
		tok = typecheck.TempAt(pos, fn, types.Types[types.TBOOL])
		take := typecheck.Call(pos, typecheck.LookupRuntime("logosArenaRetTake"), nil, false)
		takeStmt = typecheck.Stmt(ir.NewAssignStmt(pos, tok, take))
		logosArenaRetTok[fn] = tok
	}
	if !needsFrame {
		fn.Body = append(ir.Nodes{takeStmt}, fn.Body...)
		return
	}

	uintptrType := types.Types[types.TUINTPTR]
	chunk := typecheck.TempAt(pos, fn, uintptrType)
	noscan := typecheck.TempAt(pos, fn, uintptrType)
	scan := typecheck.TempAt(pos, fn, uintptrType)

	save := typecheck.Call(pos, typecheck.LookupRuntime("logosArenaMarkSave"), nil, false)
	saveStmt := typecheck.Stmt(ir.NewAssignListStmt(pos, ir.OAS2,
		[]ir.Node{chunk, noscan, scan}, []ir.Node{save}))

	restore := typecheck.Call(pos, typecheck.LookupRuntime("logosArenaMarkRestore"),
		[]ir.Node{chunk, noscan, scan}, false)
	deferStmt := ir.NewGoDeferStmt(pos, ir.ODEFER, restore)
	deferStmt.SetEsc(ir.EscNever)
	deferStmt = typecheck.Stmt(deferStmt).(*ir.GoDeferStmt)
	if call, ok := deferStmt.Call.(*ir.CallExpr); ok {
		if clo, ok := call.Fun.(*ir.ClosureExpr); ok {
			clo.SetEsc(ir.EscNone)
		}
	}

	if isCandidate {
		condDefer := ir.NewIfStmt(pos, tok, nil, []ir.Node{deferStmt})
		condChecked := typecheck.Stmt(condDefer)
		fn.Body = append(ir.Nodes{takeStmt, saveStmt, condChecked}, fn.Body...)
	} else {
		fn.Body = append(ir.Nodes{saveStmt, deferStmt}, fn.Body...)
	}
	escape.LogosArenaRecordFrameDeferred(fn)
}

func walkLogosArenaMake(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	if !buildcfg.Experiment.LogosArena {
		return nil
	}
	if base.Ctxt.Arch.Name != "amd64" {
		return nil
	}
	if ir.CurFunc == nil {
		return nil
	}

	if escape.LogosArenaConstMake(n, ir.CurFunc) {
		if total, ok := escape.LogosArenaConstBytes(n); ok {
			return walkLogosArenaConstMake(n, init, total)
		}
		return nil
	}
	if escape.LogosArenaMake(n, ir.CurFunc) {
		return walkLogosArenaDynamicMake(n, init)
	}
	if escape.LogosArenaRetMake(n, ir.CurFunc) {
		return walkLogosArenaRetMake(n, init)
	}
	return nil
}

func walkLogosArenaRetMake(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	tok := logosArenaRetTok[ir.CurFunc]
	if tok == nil {
		return nil
	}
	lenNode := safeExpr(n.Len, init)
	capExpr := n.Cap
	if capExpr != nil {
		capExpr = safeExpr(capExpr, init)
	} else {
		capExpr = lenNode
	}
	total, isConst := escape.LogosArenaConstBytes(n)
	if isConst {
		nifLen := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, lenNode, ir.NewInt(base.Pos, 0)), []ir.Node{mkcall("panicmakeslicelen", nil, init)}, nil)
		appendWalkStmt(init, nifLen)
		nifCap := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, capExpr, ir.NewInt(base.Pos, 0)), []ir.Node{mkcall("panicmakeslicecap", nil, init)}, nil)
		appendWalkStmt(init, nifCap)
		lenCap := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(lenNode, types.Types[types.TUINT64]), typecheck.Conv(capExpr, types.Types[types.TUINT64])), []ir.Node{mkcall("panicmakeslicecap", nil, init)}, nil)
		appendWalkStmt(init, lenCap)
	}
	rtyp := reflectdata.MakeSliceElemRType(base.Pos, n)
	slice := typecheck.TempAt(base.Pos, ir.CurFunc, n.Type())
	var trueBody, falseBody ir.Nodes
	if isConst && n.Type().Elem().HasPointers() {
		allocFn := typecheck.LookupRuntime("logosArenaAllocScan")
		callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], &trueBody, rtyp, ir.NewInt(base.Pos, total))
		ptrTemp := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
		appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))
		sh := ir.NewSliceHeaderExpr(base.Pos, n.Type(), ptrTemp, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
		sh.SetTypecheck(1)
		appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, slice, sh))
	} else if isConst {
		rndSize := (total + 7) &^ 7
		var ptrTemp *ir.Name
		if total <= ssagen.LogosArenaBumpMaxBytes && !base.Flag.Race {
			fn := typecheck.LookupRuntime("logosArenaBump")
			callBump := mkcall1(fn, types.Types[types.TUNSAFEPTR], &trueBody, ir.NewInt(base.Pos, rndSize))
			ptrTemp = typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
			appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, ptrTemp, callBump))
			var ifBody ir.Nodes
			allocFn := typecheck.LookupRuntime("logosArenaAlloc")
			callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], &ifBody, rtyp, ir.NewInt(base.Pos, rndSize))
			appendWalkStmt(&ifBody, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))
			var elseBody ir.Nodes
			appendWalkStmt(&elseBody, mkcall("memclrNoHeapPointers", nil, &elseBody, ptrTemp, ir.NewInt(base.Pos, rndSize)))
			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OEQ, ptrTemp, typecheck.NodNil()), ifBody, elseBody)
			appendWalkStmt(&trueBody, typecheck.Stmt(nif))
		} else {
			allocFn := typecheck.LookupRuntime("logosArenaAlloc")
			callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], &trueBody, rtyp, ir.NewInt(base.Pos, rndSize))
			ptrTemp = typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
			appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))
		}
		sh := ir.NewSliceHeaderExpr(base.Pos, n.Type(), ptrTemp, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
		sh.SetTypecheck(1)
		appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, slice, sh))
	} else {
		allocFn := typecheck.LookupRuntime("logosArenaMakeSlice")
		int64Typ := types.Types[types.TINT64]
		ptrTemp := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
		callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], &trueBody, rtyp, typecheck.Conv(lenNode, int64Typ), typecheck.Conv(capExpr, int64Typ))
		appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))
		sh := ir.NewSliceHeaderExpr(base.Pos, n.Type(), ptrTemp, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
		sh.SetTypecheck(1)
		appendWalkStmt(&trueBody, ir.NewAssignStmt(base.Pos, slice, sh))
	}
	heapPtr := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
	heapCall := mkcall("makeslice", types.Types[types.TUNSAFEPTR], &falseBody, rtyp, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
	appendWalkStmt(&falseBody, ir.NewAssignStmt(base.Pos, heapPtr, heapCall))
	hsh := ir.NewSliceHeaderExpr(base.Pos, n.Type(), heapPtr, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
	hsh.SetTypecheck(1)
	appendWalkStmt(&falseBody, ir.NewAssignStmt(base.Pos, slice, hsh))
	nif := ir.NewIfStmt(base.Pos, tok, trueBody, falseBody)
	appendWalkStmt(init, typecheck.Stmt(nif))
	return slice
}

func walkLogosArenaConstMake(n *ir.MakeExpr, init *ir.Nodes, totalSize int64) ir.Node {
	lenNode := safeExpr(n.Len, init)
	capExpr := n.Cap
	if capExpr != nil {
		capExpr = safeExpr(capExpr, init)
	} else {
		capExpr = lenNode
	}

	nifLen := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, lenNode, ir.NewInt(base.Pos, 0)), []ir.Node{mkcall("panicmakeslicelen", nil, init)}, nil)
	appendWalkStmt(init, nifLen)

	nifCap := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, capExpr, ir.NewInt(base.Pos, 0)), []ir.Node{mkcall("panicmakeslicecap", nil, init)}, nil)
	appendWalkStmt(init, nifCap)

	lenCap := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(lenNode, types.Types[types.TUINT64]), typecheck.Conv(capExpr, types.Types[types.TUINT64])), []ir.Node{mkcall("panicmakeslicecap", nil, init)}, nil)
	appendWalkStmt(init, lenCap)

	rtyp := reflectdata.MakeSliceElemRType(base.Pos, n)
	rndSize := (totalSize + 7) &^ 7

	var ptrTemp *ir.Name
	if totalSize <= ssagen.LogosArenaBumpMaxBytes && !base.Flag.Race {
		fn := typecheck.LookupRuntime("logosArenaBump")
		callBump := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, ir.NewInt(base.Pos, rndSize))
		ptrTemp = typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ptrTemp, callBump))

		var ifBody ir.Nodes
		allocFn := typecheck.LookupRuntime("logosArenaAlloc")
		callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], &ifBody, rtyp, ir.NewInt(base.Pos, rndSize))
		appendWalkStmt(&ifBody, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))

		var elseBody ir.Nodes
		appendWalkStmt(&elseBody, mkcall("memclrNoHeapPointers", nil, &elseBody, ptrTemp, ir.NewInt(base.Pos, rndSize)))

		nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OEQ, ptrTemp, typecheck.NodNil()), ifBody, elseBody)
		appendWalkStmt(init, typecheck.Stmt(nif))
	} else {
		allocFn := typecheck.LookupRuntime("logosArenaAlloc")
		callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], init, rtyp, ir.NewInt(base.Pos, rndSize))
		ptrTemp = typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))
	}

	return walkLogosArenaSlice(n, init, ptrTemp, lenNode, capExpr)
}

func walkLogosArenaDynamicMake(n *ir.MakeExpr, init *ir.Nodes) ir.Node {
	lenNode := safeExpr(n.Len, init)
	capExpr := n.Cap
	if capExpr != nil {
		capExpr = safeExpr(capExpr, init)
	} else {
		capExpr = lenNode
	}

	rtyp := reflectdata.MakeSliceElemRType(base.Pos, n)
	allocFn := typecheck.LookupRuntime("logosArenaMakeSlice")
	int64Typ := types.Types[types.TINT64]
	ptrTemp := typecheck.TempAt(base.Pos, ir.CurFunc, types.Types[types.TUNSAFEPTR])
	callAlloc := mkcall1(allocFn, types.Types[types.TUNSAFEPTR], init, rtyp, typecheck.Conv(lenNode, int64Typ), typecheck.Conv(capExpr, int64Typ))
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ptrTemp, callAlloc))

	return walkLogosArenaSlice(n, init, ptrTemp, lenNode, capExpr)
}

func walkLogosArenaSlice(n *ir.MakeExpr, init *ir.Nodes, ptr *ir.Name, lenNode, capExpr ir.Node) ir.Node {
	sh := ir.NewSliceHeaderExpr(base.Pos, n.Type(), ptr, typecheck.Conv(lenNode, types.Types[types.TINT]), typecheck.Conv(capExpr, types.Types[types.TINT]))
	sh.SetTypecheck(1)
	slice := typecheck.TempAt(base.Pos, ir.CurFunc, n.Type())
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, slice, sh))
	return slice
}
