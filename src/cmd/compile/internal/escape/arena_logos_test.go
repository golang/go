// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"
	"math/rand/v2"
	"os"
	"testing"

	"cmd/compile/internal/base"
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

func logosArenaTestMake(lenNode, capNode ir.Node, esc uint16) *ir.MakeExpr {
	n := ir.NewMakeExpr(src.NoXPos, ir.OMAKESLICE, lenNode, capNode)
	n.SetType(types.NewSlice(types.Types[types.TUINT8]))
	n.SetEsc(esc)
	return n
}

func logosArenaTestLen() ir.Node {
	return ir.NewNameAt(src.NoXPos, types.LocalPkg.Lookup("n"), types.Types[types.TINT])
}

func logosArenaDynamicMake(esc uint16) *ir.MakeExpr {
	return logosArenaTestMake(logosArenaTestLen(), nil, esc)
}

func TestLogosArenaShadowNeverReducesEscape(t *testing.T) {
	enableLogosArena(t)

	tests := []struct {
		name string
		make *ir.MakeExpr
		want bool
	}{
		{name: "non-escaping dynamic make", make: logosArenaDynamicMake(ir.EscNone), want: true},
		{name: "escaping dynamic make", make: logosArenaDynamicMake(ir.EscHeap), want: false},
		{name: "make stored in sync.Pool", make: logosArenaDynamicMake(ir.EscHeap), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LogosArenaMake(tt.make, &ir.Func{}); got != tt.want {
				t.Errorf("LogosArenaMake() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLogosArenaForeignPrecedence(t *testing.T) {
	enableLogosArena(t)

	control := logosArenaDynamicMake(ir.EscNone)
	if !foreignBacking(&location{n: control}) {
		t.Fatal("dynamic make is not foreignBacking-shaped; test would not exercise the foreign shape")
	}
	if !LogosArenaMake(control, &ir.Func{}) {
		t.Fatal("variable-capacity make without a foreign heap decision was vetoed")
	}

	foreignForced := logosArenaDynamicMake(ir.EscHeap)
	if !logosArenaSolvedHeap(foreignForced) {
		t.Fatal("EscHeap make not reported as heap-forced")
	}
	if LogosArenaMake(foreignForced, &ir.Func{}) {
		t.Error("foreign-forced make was arena-placed by LogosArenaMake")
	}
	if LogosArenaConstMake(foreignForced, &ir.Func{}) {
		t.Error("foreign-forced make was arena-placed by LogosArenaConstMake")
	}

	fn := &ir.Func{}
	b := &batch{}
	shadowForced := &location{n: foreignForced, curfn: fn, logosNoRet: true}
	shadowForced.attrs |= attrEscapes
	b.logosArena.makes = append(b.logosArena.makes, logosArenaMakeLoc{n: foreignForced, shadow: shadowForced})

	returned := logosArenaDynamicMake(ir.EscHeap)
	shadowReturned := &location{n: returned, curfn: fn, logosNoRet: true}
	b.logosArena.makes = append(b.logosArena.makes, logosArenaMakeLoc{n: returned, shadow: shadowReturned})

	b.logosArenaMark([]*ir.Func{fn})
	t.Cleanup(func() {
		logosArenaReg.mu.Lock()
		defer logosArenaReg.mu.Unlock()
		delete(logosArenaReg.retSites, foreignForced)
		delete(logosArenaReg.retSites, returned)
		logosArenaReg.recompute()
	})

	logosArenaReg.mu.RLock()
	_, forcedRegistered := logosArenaReg.retSites[foreignForced]
	_, returnedRegistered := logosArenaReg.retSites[returned]
	logosArenaReg.mu.RUnlock()
	if forcedRegistered {
		t.Error("foreign-forced returned make was recorded as a caller-region site")
	}
	if !returnedRegistered {
		t.Error("result-only returned make was not recorded as a caller-region site")
	}
	if LogosArenaRetMake(foreignForced, fn) {
		t.Error("foreign-forced make reported by LogosArenaRetMake")
	}
}

func TestLogosArenaVetoes(t *testing.T) {
	enableLogosArena(t)
	pos := src.NoXPos

	t.Run("goto", func(t *testing.T) {
		fn := &ir.Func{Body: ir.Nodes{ir.NewBranchStmt(pos, ir.OGOTO, nil)}}
		if !LogosArenaVetoed(fn) {
			t.Error("goto function not vetoed")
		}
		if LogosArenaPlaceable(fn) {
			t.Error("goto function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("goto function selected arena placement")
		}
	})

	t.Run("closure", func(t *testing.T) {
		fn := &ir.Func{OClosure: &ir.ClosureExpr{}}
		if logosArenaRetCapable(fn) {
			t.Error("closure is return-capable")
		}
	})

	t.Run("dupok", func(t *testing.T) {
		fn := &ir.Func{}
		fn.SetDupok(true)
		if logosArenaRetCapable(fn) {
			t.Error("dupok function is return-capable")
		}
	})

	t.Run("nosplit", func(t *testing.T) {
		fn := &ir.Func{Pragma: ir.Nosplit}
		if LogosArenaPlaceable(fn) {
			t.Error("nosplit function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("nosplit function selected arena placement")
		}
	})

	t.Run("wrapper", func(t *testing.T) {
		fn := &ir.Func{}
		fn.SetWrapper(true)
		if LogosArenaPlaceable(fn) {
			t.Error("wrapper function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("wrapper function selected arena placement")
		}
	})

	t.Run("defer", func(t *testing.T) {
		fn := &ir.Func{Body: ir.Nodes{ir.NewGoDeferStmt(pos, ir.ODEFER, nil)}}
		if LogosArenaPlaceable(fn) {
			t.Error("defer function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("defer function selected arena placement")
		}
		if LogosArenaConstMake(logosArenaTestMake(ir.NewInt(pos, 4096), nil, ir.EscNone), fn) {
			t.Error("defer function selected const arena placement")
		}
	})

	t.Run("compiling runtime", func(t *testing.T) {
		old := base.Flag.CompilingRuntime
		base.Flag.CompilingRuntime = true
		t.Cleanup(func() { base.Flag.CompilingRuntime = old })
		fn := &ir.Func{}
		if LogosArenaPlaceable(fn) {
			t.Error("runtime function is placeable")
		}
		if LogosArenaMake(logosArenaDynamicMake(ir.EscNone), fn) {
			t.Error("runtime function selected arena placement")
		}
		if LogosArenaConstMake(logosArenaTestMake(ir.NewInt(pos, 4096), nil, ir.EscNone), fn) {
			t.Error("runtime function selected const arena placement")
		}
	})
}

func TestLogosArenaConstProductOverflow(t *testing.T) {
	enableLogosArena(t)
	fn := &ir.Func{}
	pos := src.NoXPos

	constMake := func(elem *types.Type, capVal int64) *ir.MakeExpr {
		n := logosArenaTestMake(ir.NewInt(pos, capVal), nil, ir.EscNone)
		n.SetType(types.NewSlice(elem))
		return n
	}
	array := func(size int64) *types.Type {
		return types.NewArray(types.Types[types.TUINT8], size)
	}

	wrapElem, wrapCap := types.MaxWidth/2+1, int64(1)<<15
	if uint64(wrapElem)*uint64(wrapCap) != 1<<15 {
		t.Fatalf("premise: %d * %d does not wrap to 2^15", wrapElem, wrapCap)
	}
	zeroElem, zeroCap := types.MaxWidth/2, int64(1)<<15
	if uint64(zeroElem)*uint64(zeroCap) != 0 {
		t.Fatalf("premise: %d * %d does not wrap to 0", zeroElem, zeroCap)
	}

	tests := []struct {
		name      string
		make      *ir.MakeExpr
		wantBytes int64
		wantOK    bool
	}{
		{"byte slice", constMake(types.Types[types.TUINT8], 4096), 4096, true},
		{"largest byte slice", constMake(types.Types[types.TUINT8], LogosArenaMaxBytes), LogosArenaMaxBytes, true},
		{"one byte too large", constMake(types.Types[types.TUINT8], LogosArenaMaxBytes+1), 0, false},
		{"largest word slice", constMake(types.Types[types.TUINT64], LogosArenaMaxBytes/8), LogosArenaMaxBytes / 8 * 8, true},
		{"one word too large", constMake(types.Types[types.TUINT64], LogosArenaMaxBytes/8+1), 0, false},
		{"element of one chunk", constMake(array(LogosArenaMaxBytes), 1), LogosArenaMaxBytes, true},
		{"element past one chunk", constMake(array(LogosArenaMaxBytes+1), 1), 0, false},
		{"zero capacity", constMake(types.Types[types.TUINT8], 0), 0, false},
		{"product wraps to small", constMake(array(wrapElem), wrapCap), 0, false},
		{"product wraps to zero", constMake(array(zeroElem), zeroCap), 0, false},
		{"dynamic capacity", logosArenaDynamicMake(ir.EscNone), 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotBytes, gotOK := LogosArenaConstBytes(tt.make)
			if gotBytes != tt.wantBytes || gotOK != tt.wantOK {
				t.Errorf("LogosArenaConstBytes() = %d, %v, want %d, %v", gotBytes, gotOK, tt.wantBytes, tt.wantOK)
			}
			if got := LogosArenaConstMake(tt.make, fn); got != (tt.wantOK && tt.wantBytes > int64(base.Debug.VariableMakeThreshold)) {
				t.Errorf("LogosArenaConstMake() = %v for %d bytes (ok %v)", got, tt.wantBytes, tt.wantOK)
			}
		})
	}

	for _, tt := range []struct {
		name         string
		elemSize, n  int64
		wantFitsOnce bool
	}{
		{"wraps to small", wrapElem, wrapCap, false},
		{"wraps to zero", zeroElem, zeroCap, false},
		{"exact fit", 8, LogosArenaMaxBytes / 8, true},
		{"one past", 8, LogosArenaMaxBytes/8 + 1, false},
	} {
		if got := logosArenaConstFits(tt.n, tt.elemSize); got != tt.wantFitsOnce {
			t.Errorf("logosArenaConstFits(%d, %d) = %v (%s), want %v", tt.n, tt.elemSize, got, tt.name, tt.wantFitsOnce)
		}
	}
}

func logosArenaFreshRegistry(t *testing.T) {
	t.Helper()
	old := logosArenaReg
	logosArenaReg = newLogosArenaRegistry()
	t.Cleanup(func() { logosArenaReg = old })
}

type logosArenaOracleEdge struct {
	caller *ir.Func
	callee *ir.Func
	kind   logosArenaCallKind
	node   *ir.CallExpr
}

type logosArenaOracle struct {
	retSites map[*ir.MakeExpr]*ir.Func
	edges    []logosArenaOracleEdge
}

func newLogosArenaOracle() *logosArenaOracle {
	return &logosArenaOracle{retSites: make(map[*ir.MakeExpr]*ir.Func)}
}

func logosArenaAddRetSite(ref *logosArenaOracle, bt *batch, fn *ir.Func, m *ir.MakeExpr) {
	bt.logosArena.makes = append(bt.logosArena.makes, logosArenaMakeLoc{n: m, shadow: &location{n: m, curfn: fn, logosNoRet: true}})
	ref.retSites[m] = fn
}

func logosArenaAddLocalCall(ref *logosArenaOracle, bt *batch, caller, callee *ir.Func) *ir.CallExpr {
	return logosArenaAddCallKind(ref, bt, caller, callee, logosArenaCallLocal)
}

func logosArenaAddChainCall(ref *logosArenaOracle, bt *batch, caller, callee *ir.Func) *ir.CallExpr {
	return logosArenaAddCallKind(ref, bt, caller, callee, logosArenaCallChain)
}

func logosArenaAddCallKind(ref *logosArenaOracle, bt *batch, caller, callee *ir.Func, kind logosArenaCallKind) *ir.CallExpr {
	node := ir.NewCallExpr(src.NoXPos, ir.OCALLFUNC, logosArenaTestLen(), nil)
	res := &location{n: node, curfn: caller}
	resNoRet := &location{n: node, curfn: caller, logosNoRet: true}
	if kind == logosArenaCallChain {
		res.attrs |= attrEscapes
	}
	bt.logosArena.calls = append(bt.logosArena.calls, logosArenaCallLoc{call: node, caller: caller, callee: callee, res: res, resNoRet: resNoRet})
	ref.edges = append(ref.edges, logosArenaOracleEdge{caller: caller, callee: callee, kind: kind, node: node})
	return node
}

func logosArenaAddPlanCall(ref *logosArenaOracle, bt *batch, caller, callee *ir.Func, chain bool) *ir.CallExpr {
	if chain {
		return logosArenaAddChainCall(ref, bt, caller, callee)
	}
	return logosArenaAddLocalCall(ref, bt, caller, callee)
}

func (ref *logosArenaOracle) producing() map[*ir.Func]bool {
	producing := make(map[*ir.Func]bool)
	for _, fn := range ref.retSites {
		producing[fn] = true
	}
	for changed := true; changed; {
		changed = false
		for _, edge := range ref.edges {
			if edge.kind != logosArenaCallChain || !producing[edge.callee] || producing[edge.caller] {
				continue
			}
			producing[edge.caller] = true
			changed = true
		}
	}
	return producing
}

func (ref *logosArenaOracle) live() map[*ir.Func]bool {
	producing := ref.producing()
	live := make(map[*ir.Func]bool)
	for changed := true; changed; {
		changed = false
		for _, edge := range ref.edges {
			if live[edge.callee] || !producing[edge.callee] {
				continue
			}
			if edge.kind == logosArenaCallLocal || live[edge.caller] {
				live[edge.callee] = true
				changed = true
			}
		}
	}
	return live
}

func logosArenaWantConsent(edge logosArenaOracleEdge, live map[*ir.Func]bool) LogosArenaConsent {
	if !live[edge.callee] {
		return LogosArenaConsentNone
	}
	if edge.kind == logosArenaCallLocal {
		return LogosArenaConsentSet
	}
	if live[edge.caller] {
		return LogosArenaConsentChain
	}
	return LogosArenaConsentNone
}

func logosArenaWantSet(t *testing.T, stage, name string, got, want map[*ir.Func]bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %s set size %d, want %d", stage, name, len(got), len(want))
	}
	for fn := range want {
		if !got[fn] {
			t.Fatalf("%s: %s is missing a fn", stage, name)
		}
	}
}

func logosArenaCheckBatch(t *testing.T, ref *logosArenaOracle, fns []*ir.Func, stage string) (producing, live map[*ir.Func]bool) {
	t.Helper()
	producing, live = ref.producing(), ref.live()

	logosArenaReg.mu.RLock()
	logosArenaWantSet(t, stage, "producing", logosArenaReg.producing, producing)
	logosArenaWantSet(t, stage, "live", logosArenaReg.live, live)
	logosArenaReg.mu.RUnlock()

	for _, fn := range fns {
		if got := LogosArenaCandidate(fn); got != live[fn] {
			t.Errorf("%s: LogosArenaCandidate() = %v, want %v", stage, got, live[fn])
		}
	}
	for _, edge := range ref.edges {
		want := logosArenaWantConsent(edge, live)
		if got := LogosArenaCallConsent(edge.node); got != want {
			t.Errorf("%s: LogosArenaCallConsent() = %v, want %v", stage, got, want)
		}
	}
	for m, fn := range ref.retSites {
		if got := LogosArenaRetMake(m, fn); got != live[fn] {
			t.Errorf("%s: LogosArenaRetMake() = %v, want %v", stage, got, live[fn])
		}
	}
	return producing, live
}

func logosArenaCheckMonotone(t *testing.T, stage string, prev, next map[*ir.Func]bool, name string) {
	t.Helper()
	for fn := range prev {
		if !next[fn] {
			t.Fatalf("%s: %s set lost a fn across batches", stage, name)
		}
	}
}

type logosArenaEdgePlan struct {
	caller int
	callee int
	chain  bool
}

type logosArenaScenario struct {
	fns      []*ir.Func
	makes    []*ir.MakeExpr
	order    []int
	isolated map[int]bool
	batches  [][]int
	batchOf  []int
	edges    [][]logosArenaEdgePlan
	makeAt   [][]int
}

func logosArenaNewScenario(rng *rand.Rand, numFns int) *logosArenaScenario {
	s := &logosArenaScenario{
		fns:      make([]*ir.Func, numFns),
		makes:    make([]*ir.MakeExpr, numFns),
		order:    rng.Perm(numFns),
		isolated: map[int]bool{numFns - 1: true, numFns - 2: true},
	}
	for i := range s.fns {
		s.fns[i] = &ir.Func{}
		s.makes[i] = logosArenaDynamicMake(ir.EscHeap)
	}
	s.partitionBatches(rng)
	s.planRecursiveScc()
	s.planLateShapes()
	s.planRandomEdges(rng)
	for len(s.edges) < len(s.makeAt) {
		s.edges = append(s.edges, nil)
	}
	return s
}

func (s *logosArenaScenario) partitionBatches(rng *rand.Rand) {
	s.batchOf = make([]int, len(s.fns))
	s.batches = [][]int{{s.order[0], s.order[1]}}
	s.batchOf[s.order[0]], s.batchOf[s.order[1]] = 0, 0
	for _, fi := range s.order[2:] {
		if s.isolated[fi] || rng.IntN(3) != 0 {
			s.batchOf[fi] = len(s.batches) - 1
			s.batches[len(s.batches)-1] = append(s.batches[len(s.batches)-1], fi)
			continue
		}
		s.batches = append(s.batches, []int{fi})
		s.batchOf[fi] = len(s.batches) - 1
	}
	s.edges = make([][]logosArenaEdgePlan, len(s.batches))
	s.makeAt = make([][]int, len(s.batches))
}

func (s *logosArenaScenario) planEdge(caller, callee int, chain bool) {
	at := s.batchOf[caller]
	s.edges[at] = append(s.edges[at], logosArenaEdgePlan{caller: caller, callee: callee, chain: chain})
}

func (s *logosArenaScenario) planMake(fi, at int) {
	for len(s.makeAt) <= at {
		s.makeAt = append(s.makeAt, nil)
	}
	s.makeAt[at] = append(s.makeAt[at], fi)
}

func (s *logosArenaScenario) latestCaller() int {
	for i := len(s.order) - 1; i >= 0; i-- {
		if !s.isolated[s.order[i]] {
			return s.order[i]
		}
	}
	return s.order[0]
}

func (s *logosArenaScenario) planRecursiveScc() {
	root, pair := s.order[0], s.order[1]
	s.planMake(root, 0)
	s.planEdge(root, root, true)
	s.planEdge(root, pair, true)
	s.planEdge(pair, root, false)
}

func (s *logosArenaScenario) planLateShapes() {
	root := s.order[0]
	s.planEdge(s.latestCaller(), root, true)
	s.planMake(s.latestCaller(), len(s.batches))
}

func (s *logosArenaScenario) planRandomEdges(rng *rand.Rand) {
	lateCaller := s.latestCaller()
	for _, fi := range s.order[2:] {
		if s.isolated[fi] {
			continue
		}
		if fi != lateCaller && rng.IntN(4) != 0 {
			at := s.batchOf[fi]
			if rng.IntN(5) == 0 {
				at = len(s.batches) + rng.IntN(2)
			}
			s.planMake(fi, at)
		}
		for k := rng.IntN(3); k > 0; k-- {
			target := s.order[rng.IntN(len(s.order))]
			if s.isolated[target] {
				continue
			}
			if s.batchOf[target] > s.batchOf[fi] && rng.IntN(6) != 0 {
				continue
			}
			s.planEdge(fi, target, rng.IntN(100) < 35)
		}
	}
}

func (s *logosArenaScenario) feedBatch(b int, ref *logosArenaOracle) (*batch, []*ir.Func) {
	var bt batch
	batchFns := make([]*ir.Func, 0, len(s.edges[b])+len(s.makeAt[b]))
	for _, e := range s.edges[b] {
		if e.chain {
			logosArenaAddChainCall(ref, &bt, s.fns[e.caller], s.fns[e.callee])
		} else {
			logosArenaAddLocalCall(ref, &bt, s.fns[e.caller], s.fns[e.callee])
		}
		batchFns = append(batchFns, s.fns[e.caller])
	}
	for _, fi := range s.makeAt[b] {
		logosArenaAddRetSite(ref, &bt, s.fns[fi], s.makes[fi])
		batchFns = append(batchFns, s.fns[fi])
	}
	return &bt, batchFns
}

func (s *logosArenaScenario) feed(t *testing.T) {
	ref := newLogosArenaOracle()
	var prevProducing, prevLive map[*ir.Func]bool
	for b := range s.makeAt {
		bt, batchFns := s.feedBatch(b, ref)
		bt.logosArenaMark(batchFns)
		stage := fmt.Sprintf("batch%d", b)
		producing, live := logosArenaCheckBatch(t, ref, s.fns, stage)
		logosArenaCheckMonotone(t, stage, prevProducing, producing, "producing")
		logosArenaCheckMonotone(t, stage, prevLive, live, "live")
		prevProducing, prevLive = producing, live
	}
}

func TestLogosArenaRegistryIncrementalMatchesReference(t *testing.T) {
	enableLogosArena(t)

	for seed := uint64(1); seed <= 8; seed++ {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			logosArenaFreshRegistry(t)
			rng := rand.New(rand.NewPCG(seed, seed))
			logosArenaNewScenario(rng, 24).feed(t)
		})
	}
}

func TestLogosArenaCallConsentVisiblePerBatch(t *testing.T) {
	enableLogosArena(t)
	logosArenaFreshRegistry(t)
	ref := newLogosArenaOracle()

	leaf, mid, top, root := &ir.Func{}, &ir.Func{}, &ir.Func{}, &ir.Func{}
	leafRet, rootRet := logosArenaDynamicMake(ir.EscHeap), logosArenaDynamicMake(ir.EscHeap)

	var b0 batch
	logosArenaAddRetSite(ref, &b0, leaf, leafRet)
	b0.logosArenaMark([]*ir.Func{leaf})
	producing, live := logosArenaCheckBatch(t, ref, []*ir.Func{leaf}, "batch0")
	if !producing[leaf] || live[leaf] {
		t.Fatalf("batch0: leaf producing=%v live=%v, want produced and unconsumed", producing[leaf], live[leaf])
	}

	var b1 batch
	chainLeaf := logosArenaAddChainCall(ref, &b1, mid, leaf)
	chainRoot := logosArenaAddChainCall(ref, &b1, mid, root)
	b1.logosArenaMark([]*ir.Func{mid})
	producing, live = logosArenaCheckBatch(t, ref, []*ir.Func{mid}, "batch1")
	if !producing[mid] || live[mid] || live[leaf] {
		t.Fatalf("batch1: chain up the call graph gave mid producing=%v, mid live=%v, leaf live=%v; want producing only", producing[mid], live[mid], live[leaf])
	}
	if got := LogosArenaCallConsent(chainLeaf); got != LogosArenaConsentNone {
		t.Errorf("batch1: dormant chain consent = %v, want none", got)
	}

	var b2 batch
	localMid := logosArenaAddLocalCall(ref, &b2, top, mid)
	b2.logosArenaMark([]*ir.Func{top})
	producing, live = logosArenaCheckBatch(t, ref, []*ir.Func{top}, "batch2")
	if got := LogosArenaCallConsent(localMid); got != LogosArenaConsentSet {
		t.Errorf("batch2: local consent = %v, want set", got)
	}
	if got := LogosArenaCallConsent(chainLeaf); got != LogosArenaConsentChain {
		t.Errorf("batch2: chain consent = %v, want chain", got)
	}
	if got := LogosArenaCallConsent(chainRoot); got != LogosArenaConsentNone {
		t.Errorf("batch2: consent into an unproduced callee = %v, want none", got)
	}
	if !live[leaf] || !live[mid] || live[top] {
		t.Fatalf("batch2: leaf live=%v mid live=%v top live=%v; want leaf and mid only", live[leaf], live[mid], live[top])
	}

	var b3 batch
	logosArenaAddRetSite(ref, &b3, root, rootRet)
	b3.logosArenaMark([]*ir.Func{root})
	producing, live = logosArenaCheckBatch(t, ref, []*ir.Func{root}, "batch3")
	if got := LogosArenaCallConsent(chainRoot); got != LogosArenaConsentChain {
		t.Errorf("batch3: late root did not light the dormant chain consent, got %v", got)
	}
	if got := LogosArenaRetMake(rootRet, root); !got {
		t.Errorf("batch3: LogosArenaRetMake() = false for the newly live root")
	}
	if !live[root] || !producing[root] {
		t.Fatalf("batch3: root producing=%v live=%v, want both", producing[root], live[root])
	}
}
