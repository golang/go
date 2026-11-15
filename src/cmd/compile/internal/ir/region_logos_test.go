// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir_test

import (
	"os"
	"testing"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/buildcfg"
)

func TestMain(m *testing.M) {
	types.LocalPkg = types.NewPkg("p", "p")
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	typecheck.InitUniverse()
	os.Exit(m.Run())
}

func enableLogosArena(t *testing.T) {
	t.Helper()
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = true
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
}

func arenaPkg() *types.Pkg {
	return types.NewPkg("logos/arena", "arena")
}

func regionSig() *types.Type {
	return types.NewSignature(nil, nil, nil)
}

func regionCall(name string) *ir.CallExpr {
	fun := ir.NewNameAt(src.NoXPos, arenaPkg().Lookup(name), regionSig())
	fun.Class = ir.PFUNC
	return ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, fun, nil)
}

func openCall() *ir.CallExpr {
	return regionCall("Open")
}

func localName(name string) *ir.Name {
	n := ir.NewNameAt(src.NoXPos, types.LocalPkg.Lookup(name), types.Types[types.TINT])
	n.Class = ir.PAUTO
	return n
}

func testFunc(name string, body []ir.Node) *ir.Func {
	fn := ir.NewFunc(src.NoXPos, src.NoXPos, types.LocalPkg.Lookup(name), regionSig())
	fn.Body = body
	return fn
}

func ownerShape(o *ir.LogosRegionOwner) string {
	if o == nil {
		return "nil"
	}
	if o.Stmt == nil {
		return "funcbody"
	}
	return o.Stmt.Op().String()
}

func TestLogosRegionEnabled(t *testing.T) {
	old := buildcfg.Experiment.LogosArena
	buildcfg.Experiment.LogosArena = false
	t.Cleanup(func() { buildcfg.Experiment.LogosArena = old })
	if ir.LogosRegionsEnabled() {
		t.Fatalf("LogosRegionsEnabled() = true with LogosArena off")
	}
	enableLogosArena(t)
	if !ir.LogosRegionsEnabled() {
		t.Fatalf("LogosRegionsEnabled() = false with LogosArena set")
	}
}

func TestLogosRegionCallOp(t *testing.T) {
	cases := []struct {
		name string
		want ir.LogosRegionOp
	}{
		{"Open", ir.LogosRegionOpen},
		{"New", ir.LogosRegionNew},
		{"New[int]", ir.LogosRegionNew},
		{"New[go.shape.int_0]", ir.LogosRegionNew},
		{"Make", ir.LogosRegionMake},
		{"Make[go.shape.string_0]", ir.LogosRegionMake},
		{"Append", ir.LogosRegionAppend},
		{"Append[int]", ir.LogosRegionAppend},
		{"Close", ir.LogosRegionNone},
	}
	for _, tc := range cases {
		if got := ir.LogosRegionCallOp(regionCall(tc.name)); got != tc.want {
			t.Errorf("LogosRegionCallOp(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}

	other := types.NewPkg("logos/rt", "rt")
	fun := ir.NewNameAt(src.NoXPos, other.Lookup("Open"), regionSig())
	fun.Class = ir.PFUNC
	if got := ir.LogosRegionCallOp(ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, fun, nil)); got != ir.LogosRegionNone {
		t.Errorf("logos/rt.Open recognized as region call: %v", got)
	}

	pkgOpen := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, ir.NewOne(src.NoXPos, types.Types[types.TINT]), nil)
	if got := ir.LogosRegionCallOp(pkgOpen); got != ir.LogosRegionNone {
		t.Errorf("non-name callee recognized as region call: %v", got)
	}
	if ir.LogosRegionCallOp(nil) != ir.LogosRegionNone {
		t.Errorf("nil call recognized as region call")
	}
}

func TestLogosRegionElem(t *testing.T) {
	intT := types.Types[types.TINT]

	n := regionCall("New[int]")
	n.SetType(types.NewPtr(intT))
	if got := ir.LogosRegionElem(n); got != intT {
		t.Errorf("Elem of *T call = %v, want %v", got, intT)
	}

	m := regionCall("Make[int]")
	m.SetType(types.NewSlice(intT))
	if got := ir.LogosRegionElem(m); got != intT {
		t.Errorf("Elem of []T call = %v, want %v", got, intT)
	}

	o := openCall()
	if got := ir.LogosRegionElem(o); got != nil {
		t.Errorf("Elem of Open = %v, want nil", got)
	}
	if ir.LogosRegionElem(nil) != nil {
		t.Errorf("Elem of nil call = %v, want nil", ir.LogosRegionElem(nil))
	}
}

func TestLogosRegionNoInline(t *testing.T) {
	for _, name := range []string{"Open", "New", "New[int]", "Make", "Append"} {
		fn := ir.NewFunc(src.NoXPos, src.NoXPos, arenaPkg().Lookup(name), regionSig())
		if !ir.LogosRegionNoInline(fn) {
			t.Errorf("LogosRegionNoInline(arena.%s) = false, want true", name)
		}
	}

	caller := testFunc("callsOpen", []ir.Node{openCall()})
	if !ir.LogosRegionNoInline(caller) {
		t.Errorf("LogosRegionNoInline(body calls Open) = false, want true")
	}

	inner := ir.NewBlockStmt(src.NoXPos, []ir.Node{openCall()})
	caller = testFunc("nestedOpen", []ir.Node{inner})
	if !ir.LogosRegionNoInline(caller) {
		t.Errorf("LogosRegionNoInline(nested block calls Open) = false, want true")
	}

	cl := &ir.ClosureExpr{Func: testFunc("lit", []ir.Node{openCall()})}
	caller = testFunc("closureOpen", []ir.Node{cl})
	if ir.LogosRegionNoInline(caller) {
		t.Errorf("LogosRegionNoInline(Open only in closure body) = true, want false")
	}

	plain := testFunc("plain", []ir.Node{localName("x")})
	if ir.LogosRegionNoInline(plain) {
		t.Errorf("LogosRegionNoInline(plain body) = true, want false")
	}
}

func TestLogosRegionOwnersNil(t *testing.T) {
	if got := ir.LogosRegionOwners(nil); got != nil {
		t.Errorf("LogosRegionOwners(nil) = %v, want nil", got)
	}
	if got := ir.LogosRegionDeclOwner(nil); got != nil {
		t.Errorf("LogosRegionDeclOwner(nil) = %v, want nil", got)
	}
}

func TestLogosRegionOwnersNestedBlocks(t *testing.T) {
	innerCall := openCall()
	inner := ir.NewBlockStmt(src.NoXPos, []ir.Node{innerCall})
	outerCall := openCall()
	outer := ir.NewBlockStmt(src.NoXPos, []ir.Node{outerCall, inner})
	topCall := openCall()
	fn := testFunc("nested", []ir.Node{topCall, outer})
	owners := ir.LogosRegionOwners(fn)

	o := owners[innerCall]
	if o == nil {
		t.Fatalf("no owner for inner call")
	}
	if o.Stmt != ir.Node(inner) || o.List != &inner.List {
		t.Fatalf("inner call owner = %v, want inner block", ownerShape(o))
	}
	par := o.Parent
	if par == nil || par.Stmt != ir.Node(outer) || par.List != &outer.List {
		t.Fatalf("inner owner parent = %v, want outer block", ownerShape(par))
	}
	root := par.Parent
	if root == nil || root.Stmt != nil || root.List != &fn.Body || root.Parent != nil {
		t.Fatalf("outer owner parent = %v, want function body root", ownerShape(root))
	}

	if got := owners[outerCall]; got == nil || got.Stmt != ir.Node(outer) {
		t.Errorf("outer call owner = %v, want outer block", ownerShape(got))
	}
	if got := owners[topCall]; got == nil || got.Stmt != nil || got.List != &fn.Body {
		t.Errorf("top-level call owner = %v, want function body root", ownerShape(got))
	}

	if !root.Encloses(o) {
		t.Errorf("root does not enclose inner owner")
	}
	if !root.Encloses(par) {
		t.Errorf("root does not enclose outer owner")
	}
	if o.Encloses(root) || o.Encloses(par) {
		t.Errorf("inner owner encloses its ancestors")
	}
}

func TestLogosRegionOwnersLoopBody(t *testing.T) {
	bodyCall := openCall()
	f := ir.NewForStmt(src.NoXPos, nil, nil, nil, []ir.Node{bodyCall}, false)
	fn := testFunc("loop", []ir.Node{f})
	owners := ir.LogosRegionOwners(fn)

	o := owners[bodyCall]
	if o == nil {
		t.Fatalf("no owner for loop-body call")
	}
	if o.Stmt != ir.Node(f) || o.List != &f.Body {
		t.Fatalf("loop-body owner = %v, want for statement", ownerShape(o))
	}
	if !o.Loop {
		t.Fatalf("loop-body owner Loop = false, want true")
	}
	root := o.Parent
	if root == nil || root.Stmt != nil || root.List != &fn.Body {
		t.Fatalf("loop owner parent = %v, want function body root", ownerShape(root))
	}
}

func TestLogosRegionOwnersInitClause(t *testing.T) {
	initCall := openCall()
	r := localName("r")
	oas := ir.NewAssignStmt(src.NoXPos, r, initCall)
	r.Defn = oas
	bodyCall := openCall()
	f := ir.NewForStmt(src.NoXPos, oas, nil, nil, []ir.Node{bodyCall}, false)
	fn := testFunc("initloop", []ir.Node{f})
	owners := ir.LogosRegionOwners(fn)

	o := owners[initCall]
	if o == nil {
		t.Fatalf("no owner for init-clause call")
	}
	if o.Stmt != ir.Node(f) {
		t.Fatalf("init-clause owner = %v, want for statement", ownerShape(o))
	}
	if o.List != nil {
		t.Fatalf("init-clause owner List = %v, want nil", o.List)
	}
	if o.Loop {
		t.Fatalf("init-clause owner Loop = true, want false")
	}

	b := owners[bodyCall]
	if b == nil {
		t.Fatalf("no owner for loop-body call")
	}
	if b.Parent != o {
		t.Fatalf("loop-body owner parent = %v, want init owner", ownerShape(b.Parent))
	}
	if !o.Encloses(b) {
		t.Fatalf("init owner does not enclose loop-body owner")
	}
	if !b.Loop {
		t.Fatalf("loop-body owner under init Loop = false, want true")
	}
}

func TestLogosRegionOwnersClosure(t *testing.T) {
	litCall := openCall()
	lit := testFunc("lit", []ir.Node{litCall})
	cl := &ir.ClosureExpr{Func: lit}
	outerCall := openCall()
	fn := testFunc("closure", []ir.Node{outerCall, cl})
	owners := ir.LogosRegionOwners(fn)

	o := owners[litCall]
	if o == nil {
		t.Fatalf("no owner for closure call")
	}
	if o.Stmt != nil || o.List != &lit.Body || o.Parent != nil {
		t.Fatalf("closure call owner = %v list=%v parent=%v, want closure function level", ownerShape(o), o.List, o.Parent)
	}
	if o.Encloses(owners[outerCall]) || (owners[outerCall] != nil && owners[outerCall].Encloses(o)) {
		t.Errorf("closure level and outer owner enclose each other")
	}
}

func TestLogosRegionDeclOwner(t *testing.T) {
	topName := localName("top")
	topOas := ir.NewAssignStmt(src.NoXPos, topName, ir.NewOne(src.NoXPos, types.Types[types.TINT]))
	topName.Defn = topOas

	inCall := openCall()
	inName := localName("in")
	inOas := ir.NewAssignStmt(src.NoXPos, inName, inCall)
	inName.Defn = inOas
	inner := ir.NewBlockStmt(src.NoXPos, []ir.Node{inOas})

	initName := localName("ri")
	initCall := openCall()
	initOas := ir.NewAssignStmt(src.NoXPos, initName, initCall)
	initName.Defn = initOas
	f := ir.NewForStmt(src.NoXPos, initOas, nil, nil, nil, false)

	fn := testFunc("decls", []ir.Node{topOas, inner, f})
	decls := ir.LogosRegionDeclOwner(fn)

	if got, ok := decls[topName]; !ok || got != nil {
		t.Errorf("decl owner of top-level local = %v (present %v), want function level", ownerShape(got), ok)
	}
	if got := decls[inName]; got == nil || got.Stmt != ir.Node(inner) {
		t.Errorf("decl owner of block local = %v, want inner block", ownerShape(got))
	}
	if got := decls[initName]; got == nil || got.Stmt != ir.Node(f) || got.List != nil {
		t.Errorf("decl owner of init local = %v, want init owner", ownerShape(got))
	}

	inOwner := ir.LogosRegionOwners(fn)[inCall]
	if inOwner == nil || !inOwner.Encloses(decls[inName]) {
		t.Errorf("owner of Open does not enclose decl owner of its block local")
	}
}

func TestLogosRegionCallOpNilFun(t *testing.T) {
	call := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, nil, nil)
	if ir.LogosRegionCallOp(call) != ir.LogosRegionNone {
		t.Fatalf("nil-Fun call recognized as region call")
	}
	ret := ir.NewReturnStmt(src.NoXPos, []ir.Node{call})
	fn := testFunc("nilfun", []ir.Node{ret})
	if ir.LogosRegionNoInline(fn) {
		t.Fatalf("nil-Fun call made fn noinline")
	}
	if ir.LogosRegionOwners(fn)[call] != nil {
		t.Fatalf("nil-Fun call recorded as Open")
	}
}

func TestLogosRegionPlan(t *testing.T) {
	call := openCall()
	if ir.LogosRegions.Placed(call) {
		t.Fatalf("Placed before Place")
	}
	ir.LogosRegions.Place(call)
	if !ir.LogosRegions.Placed(call) {
		t.Fatalf("Placed false after Place")
	}
	if ir.LogosRegions.IsLive(call) {
		t.Fatalf("IsLive before Live")
	}
	ir.LogosRegions.Live(call)
	if !ir.LogosRegions.IsLive(call) {
		t.Fatalf("IsLive false after Live")
	}
}
