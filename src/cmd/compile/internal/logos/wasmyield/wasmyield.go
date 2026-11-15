// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

type unit struct {
	sym    *obj.LSym
	object string
	text   *obj.Prog
	finish func(obj.ProgAlloc)
	calls  []*obj.LSym
	hard   bool
	yield  bool
}

var (
	mu    sync.Mutex
	units []*unit
	byKey = map[string]*unit{}
)

func Active() bool { return buildcfg.GOARCH == "wasm" }

func keyOf(sym *obj.LSym) string { return sym.Name + "#" + strconv.Itoa(int(sym.ABI())) }

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

func rewrite(text *obj.Prog) {
	for p := text; p != nil; p = p.Link {
		if p.As != obj.ACALL || p.To.Type != obj.TYPE_MEM || p.To.Sym == nil {
			continue
		}
		if p.To.Sym != ir.Syms.Deferreturn && !calleeYield(p.To.Sym) {
			p.As = wasm.ACALLNORESUME
		}
	}
	continuationCalls(text)
}

func continuationCalls(text *obj.Prog) {
	if !base.Ctxt.WasmStackSwitch || base.Flag.CompilingRuntime {
		return
	}
	for p := text; p != nil; p = p.Link {
		if p.As == obj.ACALL && p.To.Type == obj.TYPE_MEM && p.To.Sym == ir.Syms.Deferreturn {
			return
		}
	}
	for p := text; p != nil; p = p.Link {
		if p.As == obj.ACALL {
			p.As = wasm.ACALLCONT
		}
	}
}

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

func objectName(sym *obj.LSym) string {
	rest, ok := strings.CutPrefix(sym.Name, types.LocalPkg.Prefix+".")
	if !ok || !facts.ValidObject(rest) {
		return ""
	}
	return rest
}
