// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"internal/buildcfg"
)

const logosPointeeTagPrefix = "escp:"

type pointeeFacts struct {
	tags  map[*ir.Name]int
	sinks map[*ir.Name]int
}

var pointeeFactsByFunc = map[*ir.Func]pointeeFacts{}

func logosPointeeActive() bool {
	return base.Logos || buildcfg.Experiment.LogosEscape
}

func (b *batch) logosParamTag(fn *ir.Func, narg int, f *types.Field) string {
	note := b.paramTag(fn, narg, f)
	if !logosPointeeActive() {
		return note
	}
	name, ok := f.Nname.(*ir.Name)
	if !ok {
		return note
	}
	target, ok := pointeeFactsOf(fn).tags[name.Canonical()]
	if !ok {
		return note
	}
	return logosPointeeTagPrefix + strconv.Itoa(target)
}

func (e *escape) logosAssignHoles(dsts, srcs []ir.Node) []hole {
	ks := e.addrs(dsts)
	var routed []bool
	if ir.LogosRegionsEnabled() {
		e.logosRegionRecordStores(dsts, srcs, ks)
		for i := range ks {
			if k, ok := e.logosRegionStoreHole(dsts[i]); ok {
				if routed == nil {
					routed = make([]bool, len(ks))
				}
				ks[i] = k
				routed[i] = true
			}
		}
	}
	if !logosPointeeActive() {
		return ks
	}
	facts := pointeeFactsOf(e.curfn)
	for i := range ks {
		if routed != nil && routed[i] {
			continue
		}
		if i >= len(srcs) {
			return ks
		}
		target, ok := pointeeStoreTarget(dsts[i])
		if !ok {
			continue
		}
		sink, isSink := facts.sinks[target]
		if !isSink {
			continue
		}
		source, ok := srcs[i].(*ir.Name)
		if !ok {
			continue
		}
		if tag, tagged := facts.tags[source.Canonical()]; tagged && tag == sink {
			ks[i] = e.discardHole()
		}
	}
	return ks
}

func (e *escape) logosTagHole(ks []hole, fns []*ir.Name, nParams int, args ir.Nodes, paramIdx int) hole {
	fallback := func() hole { return e.mergedTagHole(ks, fns, paramIdx, nParams) }
	if !logosPointeeActive() || len(fns) != 1 {
		return fallback()
	}
	target, ok := logosCallTarget(fns[0], nParams, paramIdx)
	if !ok {
		return fallback()
	}
	if target >= len(args) {
		return e.heapHole()
	}
	if k, ok := e.logosStoreHole(args[target]); ok {
		e.logosRegionRecordArg(args[paramIdx], fns[0])
		return k
	}
	return e.heapHole()
}

func logosCallTarget(fn *ir.Name, nParams, paramIdx int) (int, bool) {
	sig := fn.Type()
	if sig == nil {
		return 0, false
	}
	callParam := paramIdx - (nParams - len(sig.Params()))
	if callParam < 0 || callParam >= len(sig.Params()) {
		return 0, false
	}
	nname, ok := sig.Params()[callParam].Nname.(*ir.Name)
	if !ok || nname.Curfn == nil {
		return 0, false
	}
	if nname.Curfn.Esc() < escFuncTagged {
		target, tagged := pointeeFactsOf(nname.Curfn).tags[nname.Canonical()]
		return target, tagged
	}
	return pointeeTagTarget(sig, callParam)
}

func (e *escape) logosStoreHole(base ir.Node) (hole, bool) {
	if k, ok := e.logosRegionArgHole(base); ok {
		return k, true
	}
	addr, ok := base.(*ir.AddrExpr)
	if !ok || addr.Op() != ir.OADDR {
		return e.heapHole(), false
	}
	obj, ok := addr.X.(*ir.Name)
	if !ok || obj.Op() != ir.ONAME || obj.Class == ir.PEXTERN || obj.Canonical().Opt == nil {
		return e.heapHole(), false
	}
	if obj.Sym() == nil || obj.Sym().IsBlank() {
		return e.heapHole(), false
	}
	return e.oldLoc(obj).asHole(), true
}

func pointeeStoreTarget(dst ir.Node) (*ir.Name, bool) {
	sel, ok := dst.(*ir.SelectorExpr)
	if !ok || sel.Op() != ir.ODOTPTR {
		return nil, false
	}
	name, ok := sel.X.(*ir.Name)
	if !ok || name.Op() != ir.ONAME {
		return nil, false
	}
	return name.Canonical(), true
}

func pointeeTagTarget(sig *types.Type, param int) (int, bool) {
	if param < 0 || param >= len(sig.Params()) {
		return 0, false
	}
	payload, ok := strings.CutPrefix(sig.Params()[param].Note, logosPointeeTagPrefix)
	if !ok {
		return 0, false
	}
	target, err := strconv.Atoi(payload)
	if err != nil || target < 0 || target >= len(sig.Params()) {
		return 0, false
	}
	return target, true
}

func pointeeFactsOf(fn *ir.Func) pointeeFacts {
	if facts, ok := pointeeFactsByFunc[fn]; ok {
		return facts
	}
	facts := pointeeAnalysis(fn)
	pointeeFactsByFunc[fn] = facts
	return facts
}

type pointeeTally struct {
	refs      map[*ir.Name]int
	sources   map[*ir.Name]int
	stores    map[*ir.Name]int
	targeted  map[*ir.Name]int
	dests     map[*ir.Name]map[*ir.Name]bool
	paramAt   map[*ir.Name]int
	params    []*types.Field
	capturing bool
}

func pointeeAnalysis(fn *ir.Func) pointeeFacts {
	sig := fn.Type()
	if sig == nil || len(fn.Body) == 0 {
		return pointeeFacts{}
	}
	tally := pointeeTally{
		refs:     map[*ir.Name]int{},
		sources:  map[*ir.Name]int{},
		stores:   map[*ir.Name]int{},
		targeted: map[*ir.Name]int{},
		dests:    map[*ir.Name]map[*ir.Name]bool{},
		paramAt:  map[*ir.Name]int{},
		params:   sig.Params(),
	}
	for i, f := range tally.params {
		n, ok := f.Nname.(*ir.Name)
		if !ok || n.Type() == nil || !n.Type().IsPtr() || f.Sym == nil || f.Sym.IsBlank() {
			continue
		}
		tally.paramAt[n.Canonical()] = i
	}
	if len(tally.paramAt) < 2 {
		return pointeeFacts{}
	}
	ir.VisitList(fn.Body, tally.observe)
	if tally.capturing {
		return pointeeFacts{}
	}
	return tally.facts()
}

func (t *pointeeTally) observe(n ir.Node) {
	if pointeeCapture(n) {
		t.capturing = true
	}
	if name, ok := n.(*ir.Name); ok {
		if _, isParam := t.paramAt[name.Canonical()]; isParam {
			t.refs[name.Canonical()]++
		}
		return
	}
	target, ok := pointeeAssignTarget(n)
	if !ok {
		return
	}
	if _, isParam := t.paramAt[target]; !isParam {
		return
	}
	t.stores[target]++
	as := n.(*ir.AssignStmt)
	source, ok := as.Y.(*ir.Name)
	if !ok {
		return
	}
	source = source.Canonical()
	if _, isParam := t.paramAt[source]; !isParam {
		return
	}
	t.sources[source]++
	t.targeted[target]++
	if t.dests[source] == nil {
		t.dests[source] = map[*ir.Name]bool{}
	}
	t.dests[source][target] = true
}

func pointeeAssignTarget(n ir.Node) (*ir.Name, bool) {
	as, ok := n.(*ir.AssignStmt)
	if !ok || as.Op() != ir.OAS {
		return nil, false
	}
	return pointeeStoreTarget(as.X)
}

func (t *pointeeTally) facts() pointeeFacts {
	facts := pointeeFacts{tags: map[*ir.Name]int{}, sinks: map[*ir.Name]int{}}
	for target := range t.stores {
		if !t.sink(target) {
			continue
		}
		index := t.paramAt[target]
		facts.sinks[target] = index
		for source, dests := range t.dests {
			if dests[target] {
				facts.tags[source] = index
			}
		}
	}
	return facts
}

func (t *pointeeTally) sink(target *ir.Name) bool {
	if t.refs[target] != t.stores[target] || t.stores[target] != t.targeted[target] {
		return false
	}
	for source, dests := range t.dests {
		if !dests[target] {
			continue
		}
		if len(dests) != 1 || source == target || t.refs[source] != t.sources[source] {
			return false
		}
	}
	return true
}

func pointeeCapture(n ir.Node) bool {
	switch n.Op() {
	case ir.OCLOSURE, ir.OMETHVALUE:
		return true
	}
	return false
}
