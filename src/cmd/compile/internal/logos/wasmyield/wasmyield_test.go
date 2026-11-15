// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasmyield

import (
	"testing"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types"
	"cmd/internal/goobj"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"
)

func internalSym(name string) *obj.LSym {
	s := &obj.LSym{Name: name}
	s.SetABI(obj.ABIInternal)
	return s
}

func resetUnits(t *testing.T) {
	t.Helper()
	mu.Lock()
	units, byKey = nil, map[string]*unit{}
	mu.Unlock()
	t.Cleanup(func() {
		mu.Lock()
		units, byKey = nil, map[string]*unit{}
		mu.Unlock()
		base.LogosWasmAssume = ""
	})
}

func addUnit(name string, calls ...*obj.LSym) *unit {
	u := &unit{sym: internalSym(name), calls: calls}
	add(u)
	return u
}

func addHardUnit(name string) *unit {
	u := addUnit(name)
	u.hard = true
	return u
}

func TestSolveLeastFixpoint(t *testing.T) {
	resetUnits(t)
	c := addHardUnit("p.C")
	b := addUnit("p.B", c.sym)
	a := addUnit("p.A", b.sym)
	e := addUnit("p.E", a.sym)
	d := addUnit("p.D")
	d.calls = []*obj.LSym{d.sym}
	pure := addUnit("p.Pure")
	caller := addUnit("p.Caller", pure.sym)

	solve(units)

	want := map[*unit]bool{c: true, b: true, a: true, e: true, d: false, pure: false, caller: false}
	for u, yield := range want {
		if u.yield != yield {
			t.Errorf("%s: yield = %v, want %v", u.sym.Name, u.yield, yield)
		}
	}
}

func TestSolveUnknownCalleeYields(t *testing.T) {
	resetUnits(t)
	u := addUnit("p.F", internalSym("q.Missing"))
	solve(units)
	if !u.yield {
		t.Fatal("a call to a function with no facts must yield")
	}
}

func TestSolveDuplicateOKCalleeYields(t *testing.T) {
	resetUnits(t)
	dup := internalSym("p.G[go.shape.int]")
	dup.Set(obj.AttrDuplicateOK, true)
	add(&unit{sym: dup})
	u := addUnit("p.F", dup)
	solve(units)
	if !u.yield {
		t.Fatal("a call to a duplicate-OK function must yield")
	}
}

func TestSolveAssumedCalleeIsPure(t *testing.T) {
	resetUnits(t)
	park := addHardUnit("p.park")
	caller := addUnit("p.caller", park.sym)
	base.LogosWasmAssume = "p.park"
	solve(units)
	if !park.yield || caller.yield {
		t.Fatalf("park.yield = %v, caller.yield = %v, want true, false", park.yield, caller.yield)
	}
}

func TestABIKeysDoNotCollide(t *testing.T) {
	resetUnits(t)
	asm := &obj.LSym{Name: "p.f"}
	asm.SetABI(obj.ABI0)
	wrapper := addUnit("p.f", asm)
	solve(units)
	if !wrapper.yield {
		t.Fatal("a wrapper over an unknown ABI0 function must yield")
	}
}

func TestPureAsm(t *testing.T) {
	for name, want := range map[string]bool{
		"internal/bytealg.IndexByte": true,
		"internal/bytealg.Count":     false,
		"runtime.memmove":            true,
		"runtime.memequal":           true,
		"runtime.gogo":               false,
		"runtime.mcall":              false,
		"runtime.morestack":          false,
		"main.asmFunc":               false,
	} {
		if got := pureAsm(name); got != want {
			t.Errorf("pureAsm(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestImportYields(t *testing.T) {
	for _, c := range []struct {
		module, name string
		want         bool
	}{
		{"wasi_snapshot_preview1", "fd_write", false},
		{"wasi_snapshot_preview1", "args_get", false},
		{"wasi_snapshot_preview1", "random_get", false},
		{"wasi_snapshot_preview1", "poll_oneoff", true},
		{"wasi_snapshot_preview1", "sched_yield", true},
		{"gojs", "runtime.nanotime1", false},
		{"gojs", "runtime.walltime", false},
		{"gojs", "syscall/js.valueCall", true},
		{"gojs", "syscall/js.valueInvoke", true},
		{"unknown_module", "anything", true},
	} {
		if got := importYields(c.module, c.name); got != c.want {
			t.Errorf("importYields(%q, %q) = %v, want %v", c.module, c.name, got, c.want)
		}
	}
}

func TestSplitName(t *testing.T) {
	for _, c := range []struct {
		name, pkg, object string
		ok                bool
	}{
		{"runtime.gopark", "runtime", "gopark", true},
		{"example.com/m/p.F", "example.com/m/p", "F", true},
		{"example.com/m/p.(*T).M", "example.com/m/p", "(*T).M", true},
		{"gopkg.in/yaml%2ev3.Marshal", "gopkg.in/yaml.v3", "Marshal", true},
		{"nodot", "", "", false},
	} {
		pkg, object, ok := splitName(c.name)
		if pkg != c.pkg || object != c.object || ok != c.ok {
			t.Errorf("splitName(%q) = %q, %q, %v; want %q, %q, %v", c.name, pkg, object, ok, c.pkg, c.object, c.ok)
		}
	}
}

func TestImportedYieldReadsFacts(t *testing.T) {
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	member := facts.Encode(fp, &facts.Facts{
		GoPackage: true,
		Fns: []facts.FnFact{
			{Object: "Pure", HasMayYield: true},
			{Object: "Hot", MayYield: true, HasMayYield: true},
		},
	})
	if _, err := facts.Import("example.com/wasmyield/lib", string(fp[:]), member); err != nil {
		t.Fatalf("Import: %v", err)
	}
	for name, want := range map[string]bool{
		"example.com/wasmyield/lib.Pure":    false,
		"example.com/wasmyield/lib.Hot":     true,
		"example.com/wasmyield/lib.Missing": true,
		"example.com/wasmyield/none.Pure":   true,
	} {
		if got := importedYield(name); got != want {
			t.Errorf("importedYield(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestMergeFnsKeepsOtherFlags(t *testing.T) {
	cur := &facts.Facts{Fns: []facts.FnFact{{Object: "F", Readonly: true}}}
	got := mergeFns(cur, []facts.FnFact{
		{Object: "F", MayYield: true, HasMayYield: true},
		{Object: "G", HasMayYield: true},
	})
	if len(got) != 2 {
		t.Fatalf("got %d records, want 2: %+v", len(got), got)
	}
	if !got[0].Readonly || !got[0].MayYield || !got[0].HasMayYield {
		t.Errorf("F lost a flag: %+v", got[0])
	}
	if got[1].Object != "G" || got[1].MayYield || !got[1].HasMayYield {
		t.Errorf("G = %+v, want MayYield=0", got[1])
	}
}

func TestAddCallYieldSources(t *testing.T) {
	resetUnits(t)
	callee := internalSym("p.callee")
	u := &unit{}
	u.addCall(obj.Addr{Type: obj.TYPE_MEM, Sym: callee})
	if len(u.calls) != 1 || u.calls[0] != callee || u.hard {
		t.Fatalf("direct call: calls=%v hard=%v", u.calls, u.hard)
	}
	u.addCall(obj.Addr{Type: obj.TYPE_MEM, Sym: ir.Syms.Deferreturn})
	if !u.hard {
		t.Fatal("deferreturn must be a yield source")
	}
	v := &unit{}
	v.addCall(obj.Addr{Type: obj.TYPE_MEM})
	if !v.hard {
		t.Fatal("missing callee must be a yield source")
	}
	w := &unit{}
	w.addCall(obj.Addr{Type: obj.TYPE_NONE, Sym: callee})
	if !w.hard {
		t.Fatal("indirect call must be a yield source")
	}
}

func wasmProfileCtxt(t *testing.T) {
	t.Helper()
	prev := base.Ctxt
	base.Ctxt = &obj.Link{WasmProfile: true}
	t.Cleanup(func() { base.Ctxt = prev })
	prevPkg := types.LocalPkg
	types.LocalPkg = types.NewPkg("example.com/wasmyield/lib", "lib")
	t.Cleanup(func() { types.LocalPkg = prevPkg })
}

func TestSummarizeNilCheckYields(t *testing.T) {
	wasmProfileCtxt(t)
	nilText := &obj.Prog{From: obj.Addr{Sym: internalSym("p.nilfn")}}
	nilText.Link = &obj.Prog{As: wasm.ACALLNORESUME, To: obj.Addr{Type: obj.TYPE_MEM, Sym: ir.Syms.SigPanic}}
	if u := summarize(internalSym("p.nilfn"), nilText); !u.hard {
		t.Fatal("a nil check that can panic must be a yield source")
	}
	quiet := &obj.Prog{From: obj.Addr{Sym: internalSym("p.quiet")}}
	quiet.Link = &obj.Prog{As: wasm.ACALLNORESUME, To: obj.Addr{Type: obj.TYPE_MEM, Sym: internalSym("p.other")}}
	if u := summarize(internalSym("p.quiet"), quiet); u.hard {
		t.Fatal("a resumeless call to a known symbol must not yield by itself")
	}
}

func TestSummarizeStackCheckYields(t *testing.T) {
	wasmProfileCtxt(t)
	frame := func(offset int64, link *obj.Prog) *obj.Prog {
		p := &obj.Prog{From: obj.Addr{Sym: internalSym("p.stackfn")}}
		p.To.Offset = offset
		p.Link = link
		return p
	}
	call := &obj.Prog{As: obj.ACALL, To: obj.Addr{Type: obj.TYPE_MEM, Sym: internalSym("p.callee")}}
	if u := summarize(internalSym("p.stackfn"), frame(8, call)); !u.hard {
		t.Fatal("a prologue stack check that unwinds through morestack must be a yield source")
	}
	if u := summarize(internalSym("p.smallleaf"), frame(8, nil)); u.hard {
		t.Fatal("a small call-free leaf skips the stack check and must stay pure")
	}
	if u := summarize(internalSym("p.frameless"), frame(0, call)); u.hard {
		t.Fatal("a frameless function has no prologue check and must stay pure")
	}
}
