// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"os"
	"sync"
	"testing"

	"cmd/compile/internal/amd64"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"internal/buildcfg"
)

func TestMain(m *testing.M) {
	amd64.Init(&ssagen.Arch)
	base.Ctxt = obj.Linknew(ssagen.Arch.LinkArch)
	types.LocalPkg = types.NewPkg("p", "p")
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	typecheck.InitUniverse()
	typecheck.InitRuntime()
	regionTestSetupTypes()
	os.Exit(m.Run())
}

func enableLogosRegion(t *testing.T) {
	t.Helper()
	regionTestSetupTypes()
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = true
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
}

var regionTestPkg *types.Pkg
var regionTestType *types.Type

var regionTestSetupTypes = sync.OnceFunc(func() {
	regionTestPkg = types.NewPkg("logos/arena", "arena")
	regionTestType = types.NewStruct([]*types.Field{
		types.NewField(src.NoXPos, regionTestPkg.Lookup("h"), types.Types[types.TUNSAFEPTR]),
		types.NewField(src.NoXPos, regionTestPkg.Lookup("gen"), types.Types[types.TUINTPTR]),
	})
})

func regionTestFn() *ir.Func {
	fn := ir.NewFunc(src.NoXPos, src.NoXPos, types.LocalPkg.Lookup("testfn"), types.NewSignature(nil, nil, nil))
	fn.Dcl = []*ir.Name{}
	ir.CurFunc = fn
	return fn
}

func regionTestCall(name string, t *types.Type, args ...ir.Node) *ir.CallExpr {
	fun := ir.NewNameAt(src.NoXPos, regionTestPkg.Lookup(name), types.NewSignature(nil, nil, nil))
	fun.Class = ir.PFUNC
	call := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, fun, args)
	call.SetType(t)
	call.SetTypecheck(1)
	return call
}

func regionTestOpen(fn *ir.Func) (*ir.CallExpr, *ir.AssignStmt, *ir.Name) {
	call := regionTestCall("Open", regionTestType)
	r := typecheck.TempAt(src.NoXPos, fn, regionTestType)
	as := ir.NewAssignStmt(src.NoXPos, r, call)
	as.SetTypecheck(1)
	r.Defn = as
	return call, as, r
}

func regionTestNew(fn *ir.Func, r *ir.Name, elem *types.Type) (*ir.CallExpr, *ir.AssignStmt) {
	call := regionTestCall("New", types.NewPtr(elem), r)
	p := typecheck.TempAt(src.NoXPos, fn, types.NewPtr(elem))
	as := ir.NewAssignStmt(src.NoXPos, p, call)
	as.SetTypecheck(1)
	p.Defn = as
	return call, as
}

func regionTestBlock(stmts ...ir.Node) *ir.BlockStmt {
	blk := ir.NewBlockStmt(src.NoXPos, nil)
	blk.List = ir.Nodes(stmts)
	return blk
}

func regionCallName(n ir.Node) string {
	call, ok := n.(*ir.CallExpr)
	if !ok {
		return ""
	}
	nm, ok := call.Fun.(*ir.Name)
	if !ok {
		return ""
	}
	return nm.Sym().Name
}

func regionCloseCount(list []ir.Node) int {
	n := 0
	for _, st := range list {
		if regionCallName(st) == "logosRegionClose" {
			n++
		}
	}
	return n
}

func regionLower(t *testing.T, fn *ir.Func) {
	t.Helper()
	enableLogosRegion(t)
	logosRegionLower(fn)
}

func TestLogosRegionBumpNew(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCall, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCall)
	ir.LogosRegions.Live(openCall)
	fn.Body = ir.Nodes{regionTestBlock(openAs, newAs)}
	logosRegionLower(fn)
	blk := fn.Body[0].(*ir.BlockStmt)
	list := []ir.Node(blk.List)
	if regionCallName(openAs.Y) != "" || openAs.Y.Op() != ir.OSTRUCTLIT {
		t.Fatalf("live open not lowered to a struct literal: %v", openAs.Y.Op())
	}
	if len(openAs.Init()) == 0 || regionCallName(openAs.Init()[len(openAs.Init())-1].(*ir.AssignListStmt).Rhs[0]) != "logosRegionOpen" {
		t.Fatal("live open init does not call logosRegionOpen")
	}
	lit := openAs.Y.(*ir.CompLitExpr)
	if len(lit.List) != 2 {
		t.Fatalf("open literal has %d entries, want 2", len(lit.List))
	}
	if len(newAs.Init()) < 2 {
		t.Fatalf("bump prelude has %d statements, want at least 2", len(newAs.Init()))
	}
	bumpAs := newAs.Init()[0].(*ir.AssignStmt)
	if regionCallName(bumpAs.Y) != "logosRegionBump" {
		t.Fatalf("prelude allocates via %q, want logosRegionBump", regionCallName(bumpAs.Y))
	}
	nif := newAs.Init()[1].(*ir.IfStmt)
	if regionCallName(nif.Body[0].(*ir.AssignStmt).Y) != "logosRegionAllocNoscan" {
		t.Fatal("bump slow path does not call logosRegionAllocNoscan")
	}
	if regionCallName(nif.Else[0].(*ir.CallExpr)) != "memclrNoHeapPointers" {
		t.Fatal("bump success path does not zero the payload")
	}
	if regionCloseCount(list) != 1 {
		t.Fatalf("owner block has %d closes, want 1", regionCloseCount(list))
	}
}
func TestLogosRegionRaceDisablesBump(t *testing.T) {
	enableLogosRegion(t)
	old := base.Flag.Race
	base.Flag.Race = true
	t.Cleanup(func() { base.Flag.Race = old })
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCall, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCall)
	ir.LogosRegions.Live(openCall)
	fn.Body = ir.Nodes{regionTestBlock(openAs, newAs)}
	logosRegionLower(fn)
	for _, st := range newAs.Init() {
		if regionCallName(st) == "logosRegionBump" {
			t.Fatal("race build uses logosRegionBump")
		}
	}
	y := newAs.Y
	if conv, ok := y.(*ir.ConvExpr); ok {
		y = conv.X
	}
	if regionCallName(y) != "logosRegionAllocNoscan" {
		t.Fatalf("race new lowers to %q, want logosRegionAllocNoscan", regionCallName(y))
	}
}

func TestLogosRegionDeadOpen(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	_, deadAs, deadR := regionTestOpen(fn)
	_, deadNew := regionTestNew(fn, deadR, types.Types[types.TINT8])
	liveCall, liveAs, liveR := regionTestOpen(fn)
	liveNewCall, liveNew := regionTestNew(fn, liveR, types.Types[types.TINT8])
	ir.LogosRegions.Place(liveNewCall)
	ir.LogosRegions.Live(liveCall)
	fn.Body = ir.Nodes{regionTestBlock(deadAs, deadNew, liveAs, liveNew)}
	logosRegionLower(fn)
	blk := fn.Body[0].(*ir.BlockStmt)
	list := []ir.Node(blk.List)
	if deadAs.Y.Op() != ir.OSTRUCTLIT || len(deadAs.Y.(*ir.CompLitExpr).List) != 0 {
		t.Fatalf("dead open not lowered to Region{}: %v", deadAs.Y.Op())
	}
	if len(deadAs.Init()) != 0 {
		t.Fatal("dead open init calls logosRegionOpen")
	}
	if regionCloseCount(list) != 1 {
		t.Fatalf("owner block has %d closes, want 1 for the live record only", regionCloseCount(list))
	}
}

func TestLogosRegionScanAndSlowNew(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	scanOpen, scanAs, scanR := regionTestOpen(fn)
	scanCall, scanNew := regionTestNew(fn, scanR, types.NewPtr(types.Types[types.TINT]))
	ir.LogosRegions.Place(scanCall)
	ir.LogosRegions.Live(scanOpen)
	slowOpen, slowAs, slowR := regionTestOpen(fn)
	slowElem := types.NewArray(types.Types[types.TUINT8], 513)
	slowCall, slowNew := regionTestNew(fn, slowR, slowElem)
	ir.LogosRegions.Place(slowCall)
	ir.LogosRegions.Live(slowOpen)
	fn.Body = ir.Nodes{regionTestBlock(scanAs, scanNew, slowAs, slowNew)}
	logosRegionLower(fn)
	blk := fn.Body[0].(*ir.BlockStmt)
	list := []ir.Node(blk.List)
	scanY := scanNew.Y
	if scanY.Op() != ir.OCONVNOP && scanY.Op() != ir.OCONV {
		t.Fatalf("scan new not wrapped in a conversion: %v", scanY.Op())
	}
	conv := scanY.(*ir.ConvExpr)
	if regionCallName(conv.X) != "logosRegionAllocScan" {
		t.Fatalf("scan new lowers to %q, want logosRegionAllocScan", regionCallName(conv.X))
	}
	slowY := slowNew.Y
	convSlow := slowY.(*ir.ConvExpr)
	if regionCallName(convSlow.X) != "logosRegionAllocNoscan" {
		t.Fatalf("513-byte new lowers to %q, want logosRegionAllocNoscan", regionCallName(convSlow.X))
	}
	for _, st := range list {
		if regionCallName(st) == "logosRegionBump" {
			t.Fatal("found a logosRegionBump call without the bump gate")
		}
	}
	if regionCloseCount(list) != 2 {
		t.Fatalf("owner block has %d closes, want 2", regionCloseCount(list))
	}
}

func TestLogosRegionUnplacedStays(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	_, openAs, r := regionTestOpen(fn)
	newCall, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	fn.Body = ir.Nodes{regionTestBlock(openAs, newAs)}
	logosRegionLower(fn)
	if newAs.Y != ir.Node(newCall) {
		t.Fatalf("unplaced new rewritten to %v", newAs.Y.Op())
	}
}

func TestLogosRegionContinueCloses(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCallPlaced, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCallPlaced)
	ir.LogosRegions.Live(openCall)
	cont := ir.NewBranchStmt(src.NoXPos, ir.OCONTINUE, nil)
	blk := regionTestBlock(openAs, newAs, cont)
	forStmt := ir.NewForStmt(src.NoXPos, nil, nil, nil, nil, false)
	forStmt.Body = ir.Nodes{blk}
	fn.Body = ir.Nodes{forStmt}
	logosRegionLower(fn)
	list := []ir.Node(blk.List)
	if len(list) != 4 || regionCallName(list[2]) != "logosRegionClose" {
		t.Fatalf("continue edge has no close: %v", list)
	}
	if list[3] != ir.Node(cont) {
		t.Fatal("continue statement moved")
	}
}

func TestLogosRegionLabeledBreakCloses(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	label := types.LocalPkg.Lookup("outer")
	openCall, openAs, r := regionTestOpen(fn)
	newCallPlaced, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCallPlaced)
	ir.LogosRegions.Live(openCall)
	brk := ir.NewBranchStmt(src.NoXPos, ir.OBREAK, label)
	blk := regionTestBlock(openAs, newAs, brk)
	forStmt := ir.NewForStmt(src.NoXPos, nil, nil, nil, nil, false)
	forStmt.Label = label
	forStmt.Body = ir.Nodes{blk}
	lab := ir.NewLabelStmt(src.NoXPos, label)
	fn.Body = ir.Nodes{lab, forStmt}
	logosRegionLower(fn)
	list := []ir.Node(blk.List)
	if len(list) != 4 || regionCallName(list[2]) != "logosRegionClose" {
		t.Fatalf("labeled break edge has no close: %v", list)
	}
}

func TestLogosRegionReturnTemps(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCall, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCall)
	ir.LogosRegions.Live(openCall)
	ret := ir.NewReturnStmt(src.NoXPos, ir.Nodes{newAs.X})
	blk := regionTestBlock(openAs, newAs, ret)
	fn.Body = ir.Nodes{blk}
	logosRegionLower(fn)
	list := []ir.Node(blk.List)
	if len(list) != 5 {
		t.Fatalf("return expansion has %d statements, want 5: %v", len(list), list)
	}
	if list[0].Op() != ir.OAS || list[1].Op() != ir.OAS {
		t.Fatal("results did not move to temporaries before the closes")
	}
	if regionCallName(list[3]) != "logosRegionClose" {
		t.Fatalf("statement 3 is %v, want the close call", list[3].Op())
	}
	if list[4].Op() != ir.ORETURN {
		t.Fatalf("statement 4 is %v, want the return", list[4].Op())
	}
	if regionCloseCount(list) != 1 {
		t.Fatalf("return path has %d closes, want 1", regionCloseCount(list))
	}
}

func TestLogosRegionFallthroughCloses(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCallPlaced, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCallPlaced)
	ir.LogosRegions.Live(openCall)
	fall := ir.NewBranchStmt(src.NoXPos, ir.OFALL, nil)
	caseClause := ir.NewCaseStmt(src.NoXPos, nil, ir.Nodes{openAs, newAs, fall})
	sw := ir.NewSwitchStmt(src.NoXPos, ir.NewInt(src.NoXPos, 1), []*ir.CaseClause{caseClause})
	fn.Body = ir.Nodes{sw}
	logosRegionLower(fn)
	list := []ir.Node(caseClause.Body)
	if len(list) != 4 || regionCallName(list[2]) != "logosRegionClose" {
		t.Fatalf("fallthrough edge has no close: %v", list)
	}
	if list[3] != ir.Node(fall) {
		t.Fatal("fallthrough statement moved")
	}
}

func TestLogosRegionInitClauseCloses(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	newCallPlaced, newAs := regionTestNew(fn, r, types.Types[types.TINT8])
	ir.LogosRegions.Place(newCallPlaced)
	ir.LogosRegions.Live(openCall)
	ifStmt := ir.NewIfStmt(src.NoXPos, ir.NewBool(src.NoXPos, true), ir.Nodes{newAs}, nil)
	ifStmt.PtrInit().Prepend(openAs)
	fn.Body = ir.Nodes{ifStmt}
	logosRegionLower(fn)
	list := []ir.Node(fn.Body)
	if len(ifStmt.Init()) != 1 {
		t.Fatalf("clause close leaked into the init list: %v", ifStmt.Init())
	}
	if len(list) != 2 || regionCallName(list[1]) != "logosRegionClose" {
		t.Fatalf("init clause close missing after the statement: %v", list)
	}
}

func TestLogosRegionMakeHeader(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	ir.LogosRegions.Live(openCall)
	elem := types.NewSlice(types.Types[types.TINT])
	n := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	makeCall := regionTestCall("Make", elem, r, n)
	ir.LogosRegions.Place(makeCall)
	slice := typecheck.TempAt(src.NoXPos, fn, elem)
	makeAs := ir.NewAssignStmt(src.NoXPos, slice, makeCall)
	makeAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, makeAs)}
	logosRegionLower(fn)
	list := []ir.Node(fn.Body[0].(*ir.BlockStmt).List)
	if len(makeAs.Init()) != 2 {
		t.Fatalf("make prelude has %d statements, want 2", len(makeAs.Init()))
	}
	allocAs := makeAs.Init()[0].(*ir.AssignStmt)
	if regionCallName(allocAs.Y) != "logosRegionMakeSliceNoscan" {
		t.Fatalf("make of a pointer-free element lowers to %q, want logosRegionMakeSliceNoscan", regionCallName(allocAs.Y))
	}
	call := allocAs.Y.(*ir.CallExpr)
	if len(call.Args) != 5 {
		t.Fatalf("logosRegionMakeSliceNoscan has %d args, want 5", len(call.Args))
	}
	if call.Args[3].Type().Kind() != types.TINT64 || call.Args[4].Type().Kind() != types.TINT64 {
		t.Fatal("length is not converted to int64 at run time")
	}
	shAs := makeAs.Init()[1].(*ir.AssignStmt)
	if shAs.Y.Op() != ir.OSLICEHEADER {
		t.Fatalf("make result is %v, want a slice header", shAs.Y.Op())
	}
	if regionCloseCount(list) != 1 {
		t.Fatalf("owner block has %d closes, want 1", regionCloseCount(list))
	}
}

func TestLogosRegionMakeEntries(t *testing.T) {
	intType := types.Types[types.TINT]
	ptrType := types.NewPtr(intType)
	for _, tc := range []struct {
		name   string
		elem   *types.Type
		length func(*ir.Func) ir.Node
		first  string
	}{
		{"variable noscan", intType, func(fn *ir.Func) ir.Node { return typecheck.TempAt(src.NoXPos, fn, intType) }, "logosRegionMakeSliceNoscan"},
		{"variable scan", ptrType, func(fn *ir.Func) ir.Node { return typecheck.TempAt(src.NoXPos, fn, intType) }, "logosRegionMakeSliceScan"},
		{"constant noscan within the bump limit", intType, func(*ir.Func) ir.Node { return ir.NewInt(src.NoXPos, 8) }, "logosRegionBump"},
		{"constant noscan past the bump limit", intType, func(*ir.Func) ir.Node { return ir.NewInt(src.NoXPos, 65) }, "logosRegionMakeSliceNoscan"},
		{"constant scan", ptrType, func(*ir.Func) ir.Node { return ir.NewInt(src.NoXPos, 4) }, "logosRegionMakeSliceScan"},
		{"constant zero length", intType, func(*ir.Func) ir.Node { return ir.NewInt(src.NoXPos, 0) }, "logosRegionMakeSliceNoscan"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enableLogosRegion(t)
			fn := regionTestFn()
			openCall, openAs, r := regionTestOpen(fn)
			ir.LogosRegions.Live(openCall)
			sliceType := types.NewSlice(tc.elem)
			makeCall := regionTestCall("Make", sliceType, r, tc.length(fn))
			ir.LogosRegions.Place(makeCall)
			slice := typecheck.TempAt(src.NoXPos, fn, sliceType)
			makeAs := ir.NewAssignStmt(src.NoXPos, slice, makeCall)
			makeAs.SetTypecheck(1)
			fn.Body = ir.Nodes{regionTestBlock(openAs, makeAs)}
			logosRegionLower(fn)
			allocAs := makeAs.Init()[0].(*ir.AssignStmt)
			if got := regionCallName(allocAs.Y); got != tc.first {
				t.Fatalf("make lowers through %q, want %q", got, tc.first)
			}
		})
	}
}

func TestLogosRegionGrowEntries(t *testing.T) {
	intType := types.Types[types.TINT]
	if got := ssagen.LogosRegionGrowSliceEntry(intType); got != "logosRegionGrowSliceNoscan" {
		t.Fatalf("pointer-free element grows through %q", got)
	}
	if got := ssagen.LogosRegionGrowSliceEntry(types.NewPtr(intType)); got != "logosRegionGrowSliceScan" {
		t.Fatalf("pointer element grows through %q", got)
	}
}

func TestLogosRegionAppendRegistry(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	ir.LogosRegions.Live(openCall)
	elem := types.NewSlice(types.Types[types.TINT])
	s := typecheck.TempAt(src.NoXPos, fn, elem)
	v := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	appendCall := regionTestCall("Append", elem, r, s, v)
	ir.LogosRegions.Place(appendCall)
	appendAs := ir.NewAssignStmt(src.NoXPos, s, appendCall)
	appendAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, appendAs)}
	logosRegionLower(fn)
	app, ok := appendAs.Y.(*ir.CallExpr)
	if !ok || app.Op() != ir.OAPPEND {
		t.Fatalf("placed append not transformed to OAPPEND: %v", appendAs.Y.Op())
	}
	if len(app.Args) != 2 || app.Args[0] != ir.Node(s) {
		t.Fatal("append args lost the slice and values")
	}
	info, ok := ssagen.LogosRegionAppends.Load(app)
	if !ok {
		t.Fatal("append grow region not registered for ssagen")
	}
	grow := info.(ssagen.LogosRegionGrowInfo)
	if grow.H == nil || grow.G == nil {
		t.Fatal("registered grow region has no handle or generation")
	}
	if grow.H.Op() != ir.ODOT || grow.G.Op() != ir.ODOT {
		t.Fatalf("grow region args are %v and %v, want field selections", grow.H.Op(), grow.G.Op())
	}
}

func TestLogosRegionUnplacedAppendStays(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	_, openAs, r := regionTestOpen(fn)
	elem := types.NewSlice(types.Types[types.TINT])
	s := typecheck.TempAt(src.NoXPos, fn, elem)
	v := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	appendCall := regionTestCall("Append", elem, r, s, v)
	appendAs := ir.NewAssignStmt(src.NoXPos, s, appendCall)
	appendAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, appendAs)}
	logosRegionLower(fn)
	if appendAs.Y != ir.Node(appendCall) {
		t.Fatalf("unplaced append rewritten to %v", appendAs.Y.Op())
	}
}

func TestLogosRegionOpenNoTupleTemp(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	posBase := src.NewFileBase("region_test.go", "region_test.go")
	callPos := base.Ctxt.PosTable.XPos(src.MakePos(posBase, 10, 1))
	bogusPos := base.Ctxt.PosTable.XPos(src.MakePos(posBase, 99, 1))
	openCall, openAs, _ := regionTestOpen(fn)
	openCall.SetPos(callPos)
	openAs.SetPos(callPos)
	ir.LogosRegions.Live(openCall)
	fn.Body = ir.Nodes{regionTestBlock(openAs)}
	old := base.Pos
	base.Pos = bogusPos
	logosRegionLower(fn)
	base.Pos = old
	if len(openAs.Init()) == 0 {
		t.Fatal("open init is empty")
	}
	found := false
	for _, st := range openAs.Init() {
		as, ok := st.(*ir.AssignListStmt)
		if !ok {
			continue
		}
		if len(as.Rhs) != 1 {
			continue
		}
		rc, ok := as.Rhs[0].(*ir.CallExpr)
		if !ok {
			continue
		}
		if regionCallName(rc) != "logosRegionOpen" {
			continue
		}
		found = true
		if as.Op() != ir.OAS2FUNC {
			t.Fatalf("open tuple call uses %v, want OAS2FUNC", as.Op())
		}
		if len(as.Lhs) != 2 {
			t.Fatalf("open tuple assign has %d results, want 2", len(as.Lhs))
		}
		fields := rc.Type().Fields()
		if len(fields) != 2 {
			t.Fatalf("open call has %d results, want 2", len(fields))
		}
		for i := range 2 {
			got := as.Lhs[i].Type()
			if got.IsFuncArgStruct() {
				t.Fatalf("open temp %d has tuple type", i)
			}
			if !types.Identical(got, fields[i].Type) {
				t.Fatalf("open temp %d type mismatch", i)
			}
		}
		if rc.Pos() != callPos {
			t.Fatalf("open call pos mismatch")
		}
		if rc.Pos() == bogusPos {
			t.Fatalf("open call carries stale position")
		}
	}
	if !found {
		t.Fatal("logosRegionOpen call not found in open init")
	}
}

func TestLogosRegionDictFirstLowers(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	elem := types.Types[types.TINT]
	dict := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TUNSAFEPTR])
	newFun := ir.NewNameAt(src.NoXPos, regionTestPkg.Lookup("New[go.shape.int]"), types.NewSignature(nil, nil, nil))
	newFun.Class = ir.PFUNC
	call := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, newFun, []ir.Node{dict, r})
	call.SetType(types.NewPtr(elem))
	call.SetTypecheck(1)
	ir.LogosRegions.Place(call)
	ir.LogosRegions.Live(openCall)
	p := typecheck.TempAt(src.NoXPos, fn, types.NewPtr(elem))
	as := ir.NewAssignStmt(src.NoXPos, p, call)
	as.SetTypecheck(1)
	p.Defn = as
	fn.Body = ir.Nodes{regionTestBlock(openAs, as)}
	logosRegionLower(fn)
	if as.Y == ir.Node(call) {
		t.Fatal("dict-first placed new not lowered")
	}
	ir.Visit(as, func(n ir.Node) {
		if n == ir.Node(dict) {
			t.Fatal("lowered init keeps the dictionary argument")
		}
	})
}

func TestLogosRegionDictFirstMake(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	ir.LogosRegions.Live(openCall)
	elem := types.NewSlice(types.Types[types.TINT])
	dict := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TUNSAFEPTR])
	n := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	makeFun := ir.NewNameAt(src.NoXPos, regionTestPkg.Lookup("Make[go.shape.int]"), types.NewSignature(nil, nil, nil))
	makeFun.Class = ir.PFUNC
	makeCall := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, makeFun, []ir.Node{dict, r, n})
	makeCall.SetType(elem)
	makeCall.SetTypecheck(1)
	ir.LogosRegions.Place(makeCall)
	slice := typecheck.TempAt(src.NoXPos, fn, elem)
	makeAs := ir.NewAssignStmt(src.NoXPos, slice, makeCall)
	makeAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, makeAs)}
	logosRegionLower(fn)
	if makeAs.Y == ir.Node(makeCall) {
		t.Fatal("dict-first placed make not lowered")
	}
	ir.Visit(makeAs, func(x ir.Node) {
		if x == ir.Node(dict) {
			t.Fatal("lowered make keeps the dictionary argument")
		}
	})
}

func TestLogosRegionDictFirstAppend(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	ir.LogosRegions.Live(openCall)
	elem := types.NewSlice(types.Types[types.TINT])
	dict := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TUNSAFEPTR])
	s := typecheck.TempAt(src.NoXPos, fn, elem)
	v := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	appendFun := ir.NewNameAt(src.NoXPos, regionTestPkg.Lookup("Append[go.shape.int]"), types.NewSignature(nil, nil, nil))
	appendFun.Class = ir.PFUNC
	appendCall := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, appendFun, []ir.Node{dict, r, s, v})
	appendCall.SetType(elem)
	appendCall.SetTypecheck(1)
	ir.LogosRegions.Place(appendCall)
	appendAs := ir.NewAssignStmt(src.NoXPos, s, appendCall)
	appendAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, appendAs)}
	logosRegionLower(fn)
	app, ok := appendAs.Y.(*ir.CallExpr)
	if !ok || app.Op() != ir.OAPPEND {
		t.Fatalf("dict-first placed append not transformed to OAPPEND: %v", appendAs.Y.Op())
	}
	if len(app.Args) != 2 || app.Args[0] != ir.Node(s) {
		t.Fatal("dict-first append keeps the region or dictionary argument")
	}
}

func TestLogosRegionGrowCallConcrete(t *testing.T) {
	enableLogosRegion(t)
	fn := regionTestFn()
	openCall, openAs, r := regionTestOpen(fn)
	ir.LogosRegions.Live(openCall)
	elem := types.NewSlice(types.Types[types.TINT])
	s := typecheck.TempAt(src.NoXPos, fn, elem)
	v := typecheck.TempAt(src.NoXPos, fn, types.Types[types.TINT])
	appendCall := regionTestCall("Append", elem, r, s, v)
	ir.LogosRegions.Place(appendCall)
	appendAs := ir.NewAssignStmt(src.NoXPos, s, appendCall)
	appendAs.SetTypecheck(1)
	fn.Body = ir.Nodes{regionTestBlock(openAs, appendAs)}
	logosRegionLower(fn)
	app, _ := appendAs.Y.(*ir.CallExpr)
	if app == nil || app.Op() != ir.OAPPEND {
		t.Fatalf("placed append not transformed to OAPPEND: %v", appendAs.Y.Op())
	}
	if _, ok := ssagen.LogosRegionAppends.Load(app); !ok {
		t.Fatal("append grow region not registered for ssagen")
	}
	var init ir.Nodes
	oldPtr := ir.NewUnaryExpr(src.NoXPos, ir.OSPTR, s)
	oldPtr.SetType(types.Types[types.TUNSAFEPTR])
	oldPtr.SetTypecheck(1)
	newLen := ir.NewInt(src.NoXPos, 3)
	oldCap := ir.NewInt(src.NoXPos, 4)
	num := ir.NewInt(src.NoXPos, 1)
	made := logosRegionGrowslice(logosRegionGrow{app, s, &init, oldPtr, newLen, oldCap, num})
	if made.Type() == nil || !types.Identical(made.Type().Elem(), types.Types[types.TINT]) {
		t.Fatal("grow call does not carry the concrete element type")
	}
}
