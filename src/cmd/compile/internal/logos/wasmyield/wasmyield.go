// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package wasmyield computes the MayYield facts of the WASM profile
// (loGo/s M16-S1, SPEC Sec 14.1) and applies them: a call whose callee can
// never unwind the wasm stack is emitted as obj/wasm ACALLNORESUME, and a
// function left with no resume-point call keeps no br_table dispatch.
//
// A function may yield when it can return the unwind flag to its caller:
// it has a defer, an indirect call, a call to a function that may yield or
// has no known facts, a nil check or a panic call, a prologue stack check
// (stack growth and preemption unwind through runtime.morestack), or it is
// a pinned scheduler entry or a blocking wasm import. The other functions
// are pure.
//
// The backend records one summary per function right after the SSA code
// generator. With the profile off it assembles at once and only the facts
// are produced. With the profile on it holds the Progs until every
// top-level function of the package is compiled, because a caller's resume
// points depend on callee facts that concurrent workers produce in no fixed
// order; Finish then solves the call graph, rewrites calls to pure callees,
// assembles every function in name order, and installs the facts for the
// __.LOGOS member.
package wasmyield

import (
	"cmp"
	"internal/buildcfg"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"
)

// unit is one function of the package being compiled.
type unit struct {
	sym    *obj.LSym
	object string              // fact object name; "" when no fact is recorded
	text   *obj.Prog           // first Prog; kept only while the flush is deferred
	finish func(obj.ProgAlloc) // assembles the function; nil once flushed
	calls  []*obj.LSym         // statically called functions
	hard   bool                // a yield source other than a call
	yield  bool
}

var (
	mu    sync.Mutex
	units []*unit
	byKey = map[string]*unit{}
)

// Active reports whether the backend produces MayYield facts: every wasm
// build does, under either command.
func Active() bool { return buildcfg.GOARCH == "wasm" }

func keyOf(sym *obj.LSym) string { return sym.Name + "#" + strconv.Itoa(int(sym.ABI())) }

// Record summarizes fn after genssa. text is the first Prog of the function
// and finish assembles it. Record reports whether it took ownership of the
// assembly, in which case the caller must not assemble: Finish does.
func Record(fn *ir.Func, text *obj.Prog, finish func(obj.ProgAlloc)) bool {
	if !Active() {
		return false
	}
	u := summarize(fn.LSym, text)
	deferred := base.Ctxt.WasmProfile
	if deferred {
		u.text, u.finish = text, finish
	}
	add(u)
	return deferred
}

// RecordImport records the wasm import wrapper of fn, whose facts come from
// the import table rather than from a body.
func RecordImport(fn *ir.Func) {
	if !Active() || fn.WasmImport == nil {
		return
	}
	u := &unit{sym: fn.LSym, object: objectName(fn.LSym)}
	u.hard = importYields(fn.WasmImport.Module, fn.WasmImport.Name)
	add(u)
}

func add(u *unit) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := byKey[keyOf(u.sym)]; dup {
		return
	}
	byKey[keyOf(u.sym)] = u
	units = append(units, u)
}

// Finish ends the backend phase of the package: it solves the facts,
// rewrites and assembles the deferred functions, and installs the facts.
func Finish() {
	if !Active() {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	solve(units)
	flush(units)
	installFacts()
}

// flush rewrites calls to pure callees and assembles the deferred
// functions in name order, so object layout is reproducible.
func flush(all []*unit) {
	pending := slices.Clone(all)
	slices.SortFunc(pending, func(a, b *unit) int { return cmp.Compare(keyOf(a.sym), keyOf(b.sym)) })
	for _, u := range pending {
		if u.finish == nil {
			continue
		}
		rewrite(u.text)
		finish := u.finish
		u.text, u.finish = nil, nil
		finish(newProg)
	}
}

func newProg() *obj.Prog {
	p := new(obj.Prog)
	p.Ctxt = base.Ctxt
	return p
}

// rewrite turns each resume-point call to a pure callee into a call
// without one. A call to deferreturn always keeps its resume point.
func rewrite(text *obj.Prog) {
	for p := text; p != nil; p = p.Link {
		if p.As != obj.ACALL || p.To.Type != obj.TYPE_MEM || p.To.Sym == nil {
			continue
		}
		if p.To.Sym != ir.Syms.Deferreturn && !calleeYield(p.To.Sym) {
			p.As = wasm.ACALLNORESUME
		}
	}
}

// calleeYield reports whether the callee may yield. Callers hold mu.
func calleeYield(sym *obj.LSym) bool {
	if base.LogosWasmAssumed(sym.Name) {
		return false
	}
	if u := byKey[keyOf(sym)]; u != nil {
		return u.sym.DuplicateOK() || u.yield
	}
	if sym.ABI() == obj.ABI0 {
		return !pureAsm(sym.Name)
	}
	return importedYield(sym.Name)
}

// installFacts publishes one MayYield fact per recorded function through
// facts.Current. Duplicate-OK functions are skipped: the linker keeps one
// of several equivalent bodies, so no single fact describes them.
func installFacts() {
	var yield []facts.FnFact
	for _, u := range units {
		if u.object != "" && !u.sym.DuplicateOK() && u.sym.ABI() == obj.ABIInternal {
			yield = append(yield, facts.FnFact{Object: u.object, MayYield: u.yield, HasMayYield: true})
		}
	}
	merged := &facts.Facts{}
	cur := facts.Current()
	if cur != nil {
		*merged = *cur
	}
	merged.Fns = mergeFns(cur, yield)
	merged.GoPackage = !base.Logos
	facts.SetCurrent(merged)
}

// mergeFns folds the yield facts into the fn records other producers
// installed, so each object keeps exactly one record.
func mergeFns(cur *facts.Facts, yield []facts.FnFact) []facts.FnFact {
	var out []facts.FnFact
	if cur != nil {
		out = slices.Clone(cur.Fns)
	}
	at := make(map[string]int, len(out))
	for i, f := range out {
		at[f.Object] = i
	}
	for _, y := range yield {
		if i, ok := at[y.Object]; ok {
			out[i].MayYield, out[i].HasMayYield = y.MayYield, true
			continue
		}
		at[y.Object] = len(out)
		out = append(out, y)
	}
	return out
}

// objectName spells sym as a facts object relative to the package being
// compiled, or "" when the name is outside the facts grammar.
func objectName(sym *obj.LSym) string {
	rest, ok := strings.CutPrefix(sym.Name, types.LocalPkg.Prefix+".")
	if !ok || !facts.ValidObject(rest) {
		return ""
	}
	return rest
}
