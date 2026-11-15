// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"slices"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"internal/buildcfg"
)

const LogosArenaMaxBytes = (64 << 10) - 1024

func LogosArenaEnabled() bool {
	return buildcfg.Experiment.LogosArena
}

func logosArenaSolvedHeap(n *ir.MakeExpr) bool {
	return n != nil && n.Esc() == ir.EscHeap
}

func logosArenaShadowSolvedHeap(shadow *location) bool {
	return shadow != nil && shadow.hasAttr(attrEscapes)
}

type LogosArenaConsent uint8

const (
	LogosArenaConsentNone LogosArenaConsent = iota

	LogosArenaConsentSet

	LogosArenaConsentChain
)

type logosArenaCallKind uint8

const (
	logosArenaCallLocal logosArenaCallKind = iota + 1
	logosArenaCallChain
)

type logosArenaBatch struct {
	makes []logosArenaMakeLoc
	calls []logosArenaCallLoc
}

type logosArenaMakeLoc struct {
	n          *ir.MakeExpr
	shadow     *location
	loops      []ir.Node
	inCondPost bool
	loopDepth  int
}

type logosArenaCallLoc struct {
	call       *ir.CallExpr
	caller     *ir.Func
	callee     *ir.Func
	res        *location
	resNoRet   *location
	loops      []ir.Node
	inCondPost bool
	loopDepth  int
}

type logosArenaCallSite struct {
	caller *ir.Func
	callee *ir.Func
	kind   logosArenaCallKind
}

type logosArenaFnInfo struct {
	hasDefer bool
	hasGoto  bool
}

type logosArenaRegistry struct {
	mu sync.RWMutex

	fns     map[*ir.Func]logosArenaFnInfo
	returns map[ir.Node]bool

	frameDeferred map[*ir.Func]bool

	loopRestores map[*ir.Func]map[ir.Node]bool

	retSites map[*ir.MakeExpr]*ir.Func

	calls      []logosArenaCallSite
	callByNode map[*ir.CallExpr]int

	producing map[*ir.Func]bool
	live      map[*ir.Func]bool
}

var logosArenaReg = logosArenaRegistry{
	fns:           make(map[*ir.Func]logosArenaFnInfo),
	returns:       make(map[ir.Node]bool),
	frameDeferred: make(map[*ir.Func]bool),
	loopRestores:  make(map[*ir.Func]map[ir.Node]bool),
	retSites:      make(map[*ir.MakeExpr]*ir.Func),
	callByNode:    make(map[*ir.CallExpr]int),
	producing:     make(map[*ir.Func]bool),
	live:          make(map[*ir.Func]bool),
}

func logosArenaInfo(fn *ir.Func) logosArenaFnInfo {
	logosArenaReg.mu.RLock()
	info, ok := logosArenaReg.fns[fn]
	logosArenaReg.mu.RUnlock()
	if ok {
		return info
	}

	var returned []ir.Node
	var scan func(n ir.Node)
	scan = func(n ir.Node) {
		if n == nil {
			return
		}
		switch n := n.(type) {
		case *ir.ClosureExpr:
			return
		case *ir.ReturnStmt:
			for _, res := range n.Results {
				ir.Visit(res, func(m ir.Node) {
					if m, ok := m.(*ir.MakeExpr); ok && m.Op() == ir.OMAKESLICE {
						returned = append(returned, m)
					}
				})
			}
		case *ir.GoDeferStmt:
			if n.Op() == ir.ODEFER {
				info.hasDefer = true
			}
		case *ir.BranchStmt:
			if n.Op() == ir.OGOTO && (n.Label == nil || len(n.Label.Name) == 0 || n.Label.Name[0] != '.') {
				info.hasGoto = true
			}
		}
		ir.DoChildren(n, func(c ir.Node) bool { scan(c); return false })
	}
	scan(fn)
	logosArenaReg.store(fn, info, returned)
	return info
}

func (r *logosArenaRegistry) store(fn *ir.Func, info logosArenaFnInfo, returned []ir.Node) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fns[fn] = info
	for _, m := range returned {
		r.returns[m] = true
	}
}

func LogosArenaPlaceable(fn *ir.Func) bool {
	if !LogosArenaEnabled() || base.Flag.CompilingRuntime {
		return false
	}
	if fn.Pragma&ir.Nosplit != 0 || fn.Wrapper() {
		return false
	}
	info := logosArenaInfo(fn)
	return !info.hasDefer && !info.hasGoto
}

func LogosArenaNeedsFrame(fn *ir.Func) bool {
	if !LogosArenaPlaceable(fn) {
		return false
	}
	if LogosArenaCandidate(fn) {
		return true
	}
	found := false
	ir.VisitList(fn.Body, func(n ir.Node) {
		if found {
			return
		}
		if LogosArenaObject(n, fn) {
			found = true
			return
		}
		makeExpr, ok := n.(*ir.MakeExpr)
		if !ok || makeExpr.Op() != ir.OMAKESLICE {
			return
		}
		if LogosArenaConstMake(makeExpr, fn) || LogosArenaMake(makeExpr, fn) {
			found = true
		}
	})
	if found {
		return true
	}
	hasConsent := false
	var scan func(n ir.Node)
	scan = func(n ir.Node) {
		if n == nil || hasConsent {
			return
		}
		if _, ok := n.(*ir.ClosureExpr); ok {
			return
		}
		if call, ok := n.(*ir.CallExpr); ok && call.Op() == ir.OCALLFUNC {
			if LogosArenaCallConsent(call) != LogosArenaConsentNone {
				hasConsent = true
				return
			}
		}
		ir.DoChildren(n, func(c ir.Node) bool { scan(c); return false })
	}
	for _, stmt := range fn.Body {
		if hasConsent {
			break
		}
		scan(stmt)
	}
	return hasConsent
}

func LogosArenaRecordFrameDeferred(fn *ir.Func) {
	logosArenaReg.mu.Lock()
	logosArenaReg.frameDeferred[fn] = true
	logosArenaReg.mu.Unlock()
}

func LogosArenaFrameDeferred(fn *ir.Func) bool {
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.frameDeferred[fn]
}

func LogosArenaLoopRestorable(fn *ir.Func, loop ir.Node) bool {
	if loop == nil {
		return false
	}
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.loopRestores[fn][loop]
}

func LogosArenaTransferLoopRestore(fn *ir.Func, rng *ir.RangeStmt, nfor *ir.ForStmt) {
	if fn == nil || rng == nil || nfor == nil {
		return
	}
	logosArenaReg.mu.Lock()
	defer logosArenaReg.mu.Unlock()
	restores := logosArenaReg.loopRestores[fn]
	if restores == nil {
		return
	}
	v, ok := restores[rng]
	if !ok {
		return
	}
	restores[nfor] = v
}

func LogosArenaVetoed(fn *ir.Func) bool {
	return logosArenaInfo(fn).hasGoto
}

func logosArenaReturned(n ir.Node) bool {
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.returns[n]
}

func logosArenaRetCapable(fn *ir.Func) bool {
	return fn.OClosure == nil && !fn.Dupok() && LogosArenaPlaceable(fn)
}

func (b *batch) logosArenaRetOutlives(l, other *location) bool {
	if l.curfn != other.curfn {
		return true
	}
	return l.loopDepth < other.loopDepth
}

func (e *escape) logosArenaMake(k hole, n *ir.MakeExpr, _ hole) {
	if !LogosArenaEnabled() {
		return
	}
	if k.dst == &e.blankLoc || !logosArenaRetCapable(e.curfn) {
		return
	}
	shadow := &location{n: n, curfn: e.curfn, loopDepth: e.loopDepth, logosNoRet: true}
	e.flow(k.addr(n, "spill"), shadow)
	e.logosArena.makes = append(e.logosArena.makes, logosArenaMakeLoc{
		n: n, shadow: shadow, loops: slices.Clone(e.loopStack), inCondPost: e.logosArenaCondPost > 0, loopDepth: e.loopDepth,
	})
}

func (e *escape) logosArenaCall(ks []hole, call *ir.CallExpr, fns []*ir.Name) {
	if ks == nil || len(fns) != 1 || !LogosArenaEnabled() {
		return
	}
	if call.Op() != ir.OCALLFUNC || !LogosArenaPlaceable(e.curfn) {
		return
	}
	callee := logosArenaStaticCallee(call.Fun)
	if callee == nil || callee.Nname != fns[0] {
		return
	}
	results := callee.Type().Results()
	for i, k := range ks {
		if i < len(results) && results[i].Type.HasPointers() && k.dst != &e.blankLoc && k.derefs < 0 {
			return
		}
	}
	var res, resNoRet *location
	for i, k := range ks {
		if i >= len(results) || !results[i].Type.HasPointers() || k.dst == &e.blankLoc {
			continue
		}
		if res == nil {
			res = &location{n: call, curfn: e.curfn, loopDepth: e.loopDepth}
			resNoRet = &location{n: call, curfn: e.curfn, loopDepth: e.loopDepth, logosNoRet: true}
		}
		e.flow(k.addr(call, "arena-return"), res)
		e.flow(k.addr(call, "arena-return"), resNoRet)
	}
	if res == nil {
		return
	}
	e.logosArena.calls = append(e.logosArena.calls, logosArenaCallLoc{call: call, caller: e.curfn, callee: callee, res: res, resNoRet: resNoRet, loops: slices.Clone(e.loopStack), inCondPost: e.logosArenaCondPost > 0, loopDepth: e.loopDepth})
}

func logosArenaStaticCallee(fun ir.Node) *ir.Func {
	switch fun.Op() {
	case ir.ONAME, ir.OMETHEXPR:
	default:
		return nil
	}
	name := ir.StaticCalleeName(fun)
	if name == nil || name.Class != ir.PFUNC || name.Func == nil || name.Func.OClosure != nil || name.Func.Dupok() {
		return nil
	}
	return name.Func
}

func logosArenaProducing(fn *ir.Func) bool {
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.producing[fn]
}

func logosArenaVetoLoopsLocked(fn *ir.Func, loops []ir.Node) {
	restores := logosArenaReg.loopRestores[fn]
	if restores == nil {
		restores = make(map[ir.Node]bool)
		logosArenaReg.loopRestores[fn] = restores
	}
	for _, loop := range loops {
		restores[loop] = false
	}
}

func (b *batch) logosArenaMark(fns []*ir.Func) {
	if !LogosArenaEnabled() {
		return
	}
	for _, fn := range fns {
		logosArenaInfo(fn)
	}

	placed := make(map[*ir.MakeExpr]bool)
	for i := range b.logosArena.makes {
		m := &b.logosArena.makes[i]
		if len(m.loops) == 0 {
			continue
		}
		placed[m.n] = LogosArenaConstMake(m.n, m.shadow.curfn) || LogosArenaMake(m.n, m.shadow.curfn)
	}

	logosArenaReg.mu.Lock()
	defer logosArenaReg.mu.Unlock()

	for _, fn := range fns {
		delete(logosArenaReg.loopRestores, fn)
	}
	for _, m := range b.logosArena.makes {
		if len(m.loops) == 0 {
			continue
		}
		fn := m.shadow.curfn
		restores := logosArenaReg.loopRestores[fn]
		if restores == nil {
			restores = make(map[ir.Node]bool)
			logosArenaReg.loopRestores[fn] = restores
		}
		innermost := m.loops[len(m.loops)-1]
		if _, seen := restores[innermost]; !seen && placed[m.n] {
			restores[innermost] = true
		}
		if m.inCondPost || logosArenaReg.returns[m.n] || m.n.Esc() != ir.EscNone ||
			logosArenaShadowSolvedHeap(m.shadow) || !logosArenaLocalShape(m.n) {
			logosArenaVetoLoopsLocked(fn, m.loops)
		}
	}
	for _, c := range b.logosArena.calls {
		if len(c.loops) == 0 {
			continue
		}
		if c.inCondPost {
			logosArenaVetoLoopsLocked(c.caller, c.loops)
			continue
		}
		if logosArenaShadowSolvedHeap(c.resNoRet) {
			continue
		}
		kind := logosArenaCallLocal
		if c.res.hasAttr(attrEscapes) {
			if !logosArenaRetCapableLocked(c.caller) {
				continue
			}
			kind = logosArenaCallChain
		}
		if kind == logosArenaCallChain {
			logosArenaVetoLoopsLocked(c.caller, c.loops)
			continue
		}
		if c.res.hasAttr(attrEscapes) || c.resNoRet.hasAttr(attrEscapes) {
			logosArenaVetoLoopsLocked(c.caller, c.loops)
			continue
		}
		restores := logosArenaReg.loopRestores[c.caller]
		if restores == nil {
			restores = make(map[ir.Node]bool)
			logosArenaReg.loopRestores[c.caller] = restores
		}
		innermost := c.loops[len(c.loops)-1]
		if _, seen := restores[innermost]; !seen {
			restores[innermost] = true
		}
	}

	for _, m := range b.logosArena.makes {
		fn := m.shadow.curfn
		if !logosArenaSolvedHeap(m.n) || logosArenaShadowSolvedHeap(m.shadow) || !logosArenaRetShape(m.n) {
			continue
		}
		logosArenaReg.retSites[m.n] = fn
	}
	for _, c := range b.logosArena.calls {
		if logosArenaShadowSolvedHeap(c.resNoRet) {
			continue
		}
		kind := logosArenaCallLocal
		if c.res.hasAttr(attrEscapes) {
			if !logosArenaRetCapableLocked(c.caller) {
				continue
			}
			kind = logosArenaCallChain
		}
		logosArenaReg.callByNode[c.call] = len(logosArenaReg.calls)
		logosArenaReg.calls = append(logosArenaReg.calls, logosArenaCallSite{caller: c.caller, callee: c.callee, kind: kind})
	}
	logosArenaReg.recompute()
}

func logosArenaRetCapableLocked(fn *ir.Func) bool {
	if fn.OClosure != nil || fn.Dupok() || !LogosArenaEnabled() || base.Flag.CompilingRuntime {
		return false
	}
	if fn.Pragma&ir.Nosplit != 0 || fn.Wrapper() {
		return false
	}
	info := logosArenaReg.fns[fn]
	return !info.hasDefer && !info.hasGoto
}

func logosArenaRetShape(n *ir.MakeExpr) bool {
	if HeapAllocReason(n) != "" {
		return false
	}
	elem := n.Type().Elem()
	if elem.Size() <= 0 || elem.Alignment() > 8 {
		return false
	}
	capExpr := n.Cap
	if capExpr == nil {
		capExpr = n.Len
	}
	if capExpr != nil && ir.IsSmallIntConst(capExpr) && !logosArenaConstFits(ir.Int64Val(capExpr), elem.Size()) {
		return false
	}
	return true
}

func logosArenaConstFits(n, elemSize int64) bool {
	return n <= LogosArenaMaxBytes/elemSize
}

func LogosArenaConstBytes(n *ir.MakeExpr) (int64, bool) {
	capExpr := n.Cap
	if capExpr == nil {
		capExpr = n.Len
	}
	if capExpr == nil || !ir.IsSmallIntConst(capExpr) {
		return 0, false
	}
	elemSize := n.Type().Elem().Size()
	capVal := ir.Int64Val(capExpr)
	if elemSize <= 0 || capVal <= 0 || !logosArenaConstFits(capVal, elemSize) {
		return 0, false
	}
	return capVal * elemSize, true
}

func (r *logosArenaRegistry) recompute() {
	producing := make(map[*ir.Func]bool)
	for _, fn := range r.retSites {
		producing[fn] = true
	}
	for changed := true; changed; {
		changed = false
		for _, c := range r.calls {
			if c.kind == logosArenaCallChain && producing[c.callee] && !producing[c.caller] {
				producing[c.caller] = true
				changed = true
			}
		}
	}
	live := make(map[*ir.Func]bool)
	for changed := true; changed; {
		changed = false
		for _, c := range r.calls {
			if live[c.callee] || !producing[c.callee] {
				continue
			}
			if c.kind == logosArenaCallLocal || live[c.caller] {
				live[c.callee] = true
				changed = true
			}
		}
	}
	r.producing = producing
	r.live = live
}

func LogosArenaCandidate(fn *ir.Func) bool {
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.live[fn]
}

func LogosArenaCallConsent(call *ir.CallExpr) LogosArenaConsent {
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	i, ok := logosArenaReg.callByNode[call]
	if !ok {
		return LogosArenaConsentNone
	}
	c := logosArenaReg.calls[i]
	if !logosArenaReg.live[c.callee] {
		return LogosArenaConsentNone
	}
	switch c.kind {
	case logosArenaCallLocal:
		return LogosArenaConsentSet
	case logosArenaCallChain:
		if logosArenaReg.live[c.caller] {
			return LogosArenaConsentChain
		}
	}
	return LogosArenaConsentNone
}

func LogosArenaRetMake(n *ir.MakeExpr, fn *ir.Func) bool {
	if !LogosArenaPlaceable(fn) || n.Op() != ir.OMAKESLICE {
		return false
	}
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaReg.retSites[n] == fn && logosArenaReg.live[fn]
}

func logosArenaLocalShape(n *ir.MakeExpr) bool {
	if n.Op() != ir.OMAKESLICE || n.Esc() != ir.EscNone {
		return false
	}
	elem := n.Type().Elem()
	return elem.Size() > 0 && elem.Alignment() <= 8
}

func LogosArenaConstMake(n *ir.MakeExpr, fn *ir.Func) bool {
	if !LogosArenaPlaceable(fn) || !logosArenaLocalShape(n) {
		return false
	}
	if n.Type().Elem().HasPointers() {
		return false
	}
	total, ok := LogosArenaConstBytes(n)
	if !ok || total <= int64(base.Debug.VariableMakeThreshold) {
		return false
	}
	return !logosArenaReturned(n)
}

func LogosArenaMake(n *ir.MakeExpr, fn *ir.Func) bool {
	if !LogosArenaPlaceable(fn) || !logosArenaLocalShape(n) {
		return false
	}
	capExpr := n.Cap
	if capExpr == nil {
		capExpr = n.Len
	}
	if capExpr == nil || ir.IsSmallIntConst(capExpr) {
		return false
	}
	return !logosArenaReturned(n)
}
