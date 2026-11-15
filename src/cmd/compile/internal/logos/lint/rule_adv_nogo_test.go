// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"

	"internal/testenv"
)

func checkSyncPath(t testing.TB, name, src string) *Input {
	t.Helper()
	var errs []error
	var reported []Reported
	file, _ := syntax.Parse(syntax.NewFileBase(name), strings.NewReader(src), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches)
	if len(errs) > 0 || file == nil {
		t.Fatalf("parse %s: %v", name, errs)
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Implicits:          make(map[syntax.Node]types2.Object),
		Scopes:             make(map[syntax.Node]*types2.Scope),
		Instances:          make(map[*syntax.Name]types2.Instance),
	}
	conf := &types2.Config{
		Importer: &lintImporter{pkgs: make(map[string]*types2.Package)},
		Error: func(err error) {
			if r, ok := ReportedOf(err); ok {
				reported = append(reported, r)
				return
			}
			errs = append(errs, err)
		},
	}
	pkg, _ := conf.Check("logos/sync", []*syntax.File{file}, info)
	if len(errs) > 0 {
		t.Fatalf("check %s: %v", name, errs)
	}
	return &Input{
		Pkg:      pkg,
		Info:     info,
		Files:    []*syntax.File{file},
		Sources:  map[*syntax.File][]byte{file: []byte(src)},
		Kind:     KindGo,
		Reported: reported,
	}
}

const advLeakHead = `package sync

import stdsync "sync"

type Locked[T any] struct {
	mu stdsync.Mutex
	v  T
}

func (l *Locked[T]) WithLock(f func(*T)) {
	l.mu.Lock()
	defer l.mu.Unlock()
	f(&l.v)
}

type Box struct {
	n int
	p *int
}

var leaked *Box
var leakedInt *int
var pslot **Box

func useBox(b *Box) {}
`

func TestAdvWithlockLeak(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"namedCallbackStoresParam",
			"func leakNamed(b *Box) { leaked = b }\nfunc p1(l *Locked[Box]) { l.WithLock(leakNamed) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"methodValueStoresParam",
			"func (r *Box) store(x *Box) { leaked = x }\nfunc p8(l *Locked[Box], v *Box) { l.WithLock(v.store) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"shadowedParamNameStaysClean",
			"func p3(l *Locked[Box]) { l.WithLock(func(b *Box) { func() { b := &Box{}; leaked = b }() }) }\n",
			nil,
		},
		{
			"aliasThroughLocal",
			"func p2(l *Locked[Box]) { l.WithLock(func(b *Box) { c := b; leaked = c }) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"interiorPointerReturnEscapes",
			"func p6(l *Locked[Box]) *int { var out *int; l.WithLock(func(b *Box) { v := *b; out = v.p }); return out }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"reassignedAliasStaysClean",
			"func p12(l *Locked[Box]) { l.WithLock(func(b *Box) { c := b; c = &Box{}; leaked = c }) }\n",
			nil,
		},
		{
			"useBeforeReassignReported",
			"func p13(l *Locked[Box]) { l.WithLock(func(b *Box) { c := b; leaked = c; c = &Box{} }) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"reassignInOtherBlockReported",
			"func p14(l *Locked[Box]) { l.WithLock(func(b *Box) { c := b; if b.n > 0 { c = &Box{} }; leaked = c }) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"derefStoreEscapes",
			"func p5(l *Locked[Box]) { l.WithLock(func(b *Box) { *pslot = b }) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"localDerefStoreStaysClean",
			"func p11(l *Locked[Box]) { l.WithLock(func(b *Box) { var tmp *Box; ptmp := &tmp; *ptmp = b }) }\n",
			nil,
		},
		{
			"addrOfFieldEscapes",
			"func p7(l *Locked[Box]) { l.WithLock(func(b *Box) { leakedInt = &b.n }) }\n",
			[]string{"26 correctness::withlock_leak deny"},
		},
		{
			"deferredCallArgStaysClean",
			"func p4(l *Locked[Box]) { l.WithLock(func(b *Box) { defer useBox(b) }) }\n",
			nil,
		},
		{
			"fieldReadStaysClean",
			"func p9(l *Locked[Box]) int { n := 0; l.WithLock(func(b *Box) { n = b.n }) ; return n }\n",
			nil,
		},
		{
			"valueCopyStaysClean",
			"func p10(l *Locked[Box]) int { n := 0; l.WithLock(func(b *Box) { v := *b; n = v.n }) ; return n }\n",
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSyncPath(t, "x.go", advLeakHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

const advDeadlockHead = `package sync

import stdsync "sync"

type Locked[T any] struct {
	mu stdsync.Mutex
	v  T
}

func (l *Locked[T]) WithLock(f func(*T)) {
	l.mu.Lock()
	defer l.mu.Unlock()
	f(&l.v)
}

type Emb struct {
	Locked[int]
}

type Ctr struct {
	mu Locked[int]
}

type Wrap2 struct {
	inner Ctr
}

func (w *Wrap2) lockInner() {
	w.inner.mu.WithLock(func(m *int) { *m++ })
}

`

func TestAdvReentrantDeadlock(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"aliasReceiverDeadlocks",
			"func d1(l Locked[int]) { p := &l; l.WithLock(func(n *int) { p.WithLock(func(m *int) { *m++ }) }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"embeddedFieldDeadlocks",
			"func d2(e *Emb) { e.WithLock(func(n *int) { e.Locked.WithLock(func(m *int) { *m++ }) }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"indexReceiverDeadlocks",
			"func d4(ls []*Locked[int]) { ls[0].WithLock(func(n *int) { ls[0].WithLock(func(m *int) { *m++ }) }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"indexVariableReceiverDeadlocks",
			"func e5(ls []*Locked[int], i int) { ls[i].WithLock(func(n *int) { ls[i].WithLock(func(m *int) { *m++ }) }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"differentIndexStaysClean",
			"func e6(ls []*Locked[int]) { ls[0].WithLock(func(n *int) { ls[1].WithLock(func(m *int) { *m++ }) }) }\n",
			nil,
		},
		{
			"storedLiteralCalledDeadlocks",
			"func d3(l *Locked[int]) { l.WithLock(func(n *int) { inner := func() { l.WithLock(func(m *int) { *m++ }) }; inner() }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"genericHelperDeadlocks",
			"func relock[T any](l *Locked[T]) { l.WithLock(func(n *T) {}) }\nfunc d5(l *Locked[int]) { l.WithLock(func(n *int) { relock(l) }) }\n",
			[]string{"33 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"fourDeepChainDeadlocks",
			"func c1(l *Locked[int]) { l.WithLock(func(n *int) { *n++ }) }\nfunc c2(l *Locked[int]) { c1(l) }\nfunc c3(l *Locked[int]) { c2(l) }\nfunc c4(l *Locked[int]) { c3(l) }\nfunc d6(l *Locked[int]) { l.WithLock(func(n *int) { c4(l) }) }\n",
			[]string{"36 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"directNestedDeadlocks",
			"func d12(l *Locked[int]) { l.WithLock(func(n *int) { l.WithLock(func(m *int) { *m++ }) }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"wrapperPathDeadlocks",
			"func d13(w *Wrap2) { w.inner.mu.WithLock(func(n *int) { w.lockInner() }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"differentLockStaysClean",
			"func e1(a, b *Locked[int]) { a.WithLock(func(n *int) { b.WithLock(func(m *int) { *m++ }) }) }\n",
			nil,
		},
		{
			"sequentialCallsStayClean",
			"func e2(l *Locked[int]) { l.WithLock(func(n *int) { *n++ }); l.WithLock(func(n *int) { *n++ }) }\n",
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSyncPath(t, "x.go", advDeadlockHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAdvReentrantDeadlockLimitations(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"unreachableBranchReported",
			"func d11(l *Locked[int]) { l.WithLock(func(n *int) { if false { l.WithLock(func(m *int) { *m++ }) } }) }\n",
			[]string{"32 concurrency::reentrant_withlock_deadlock deny"},
		},
		{
			"goSpawnedRelockUnreported",
			"func d14(l *Locked[int]) { l.WithLock(func(n *int) { go func() { l.WithLock(func(m *int) { *m++ }) }() }) }\n",
			nil,
		},
		{
			"selfReferentialAliasTerminates",
			"type Box2 struct { next *Box2 }\nfunc cyc(p **Box2, l *Locked[int]) { p = &(*p).next; l.WithLock(func(n *int) { *n++ }) }\n",
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSyncPath(t, "x.go", advDeadlockHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

const advLocksHead = `package p

import "sync"

type stats struct {
	mu sync.RWMutex

	//logos:guardedby mu
	total int
}

type emb struct {
	sync.Mutex

	//logos:guardedby Mutex
	n int
}

type pair struct {
	mu sync.Mutex

	//logos:guardedby mu
	n int
}

var arr [4]pair

func apply(f func() int) int { return f() }
`

func TestAdvGuardedAccess(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"gotoSkipsLockReported",
			"func g1(s *stats) int {\n goto End\n s.mu.Lock()\nEnd:\n return s.total\n}\n",
			[]string{"33 concurrency::unguarded_access forbid"},
		},
		{
			"immediateLiteralReadsHeldLock",
			"func g2(s *stats) int {\n s.mu.Lock()\n defer s.mu.Unlock()\n return func() int { return s.total }()\n}\n",
			nil,
		},
		{
			"methodValueUnlockDropsLock",
			"func g3(s *stats) int {\n s.mu.Lock()\n unlock := s.mu.Unlock\n _ = s.total\n unlock()\n return s.total\n}\n",
			[]string{"34 concurrency::unguarded_access forbid"},
		},
		{
			"promotedMethodHoldsLock",
			"func g5(e *emb) int {\n e.Lock()\n defer e.Unlock()\n return e.n\n}\n",
			nil,
		},
		{
			"lockUnlockSequenceStaysClean",
			"func h0(s *stats) int {\n s.mu.Lock()\n defer s.mu.Unlock()\n return s.total\n}\n",
			nil,
		},
		{
			"readUnderReadLockStaysClean",
			"func h1(s *stats) int {\n s.mu.RLock()\n defer s.mu.RUnlock()\n return s.total\n}\n",
			nil,
		},
		{
			"earlyReturnReadsHeldLock",
			"func g8(s *stats, c bool) int {\n s.mu.Lock()\n if c {\n return s.total\n }\n s.mu.Unlock()\n return 0\n}\n",
			nil,
		},
		{
			"loopBodyHoldsLock",
			"func g9(s *stats) {\n for i := 0; i < 3; i++ {\n  s.mu.Lock()\n  s.total++\n  s.mu.Unlock()\n }\n}\n",
			nil,
		},
		{
			"tryLockSuccessBranchStaysClean",
			"func t1(s *stats) {\n if s.mu.TryLock() {\n  _ = s.total\n  s.mu.Unlock()\n }\n}\n",
			nil,
		},
		{
			"tryLockNegatedElseStaysClean",
			"func t2(s *stats) {\n if !s.mu.TryLock() {\n } else {\n  _ = s.total\n  s.mu.Unlock()\n }\n}\n",
			nil,
		},
		{
			"tryLockStoredResultReported",
			"func t3(s *stats) {\n ok := s.mu.TryLock()\n if ok {\n  _ = s.total\n  s.mu.Unlock()\n }\n}\n",
			[]string{"32 concurrency::unguarded_access forbid"},
		},
		{
			"tryLockIgnoredResultReported",
			"func t4(s *stats) {\n s.mu.TryLock()\n _ = s.total\n}\n",
			[]string{"31 concurrency::unguarded_access forbid"},
		},
		{
			"indexLockStaysClean",
			"func i1() {\n arr[0].mu.Lock()\n arr[0].n = 1\n arr[0].mu.Unlock()\n}\n",
			nil,
		},
		{
			"indexLockVarStaysClean",
			"func i2(i int) {\n arr[i].mu.Lock()\n arr[i].n = 2\n arr[i].mu.Unlock()\n}\n",
			nil,
		},
		{
			"indexLockOtherElementReported",
			"func i3() {\n arr[0].mu.Lock()\n arr[1].n = 1\n arr[0].mu.Unlock()\n}\n",
			[]string{"31 concurrency::unguarded_access forbid"},
		},
		{
			"indexLockOtherVarReported",
			"func i4(i, j int) {\n arr[i].mu.Lock()\n arr[j].n = 3\n}\n",
			[]string{"31 concurrency::unguarded_access forbid"},
		},
		{
			"backwardGotoDeferredClosureReportsOnce",
			"func bd(s *stats, again bool) {\n defer func() { _ = s.total }()\nL:\n if again {\n  goto L\n }\n}\n",
			[]string{"30 concurrency::unguarded_access forbid"},
		},
		{
			"backwardGotoReentryReports",
			"func b1(s *stats, again bool) {\n s.mu.Lock()\nL:\n _ = s.total\n s.mu.Unlock()\n if again {\n  goto L\n }\n}\n",
			[]string{"32 concurrency::unguarded_access forbid"},
		},
		{
			"backwardGotoRelockedStaysClean",
			"func b2(s *stats, again bool) {\n s.mu.Lock()\nL:\n _ = s.total\n s.mu.Unlock()\n if again {\n  s.mu.Lock()\n  goto L\n }\n}\n",
			nil,
		},
		{
			"fiveLabelClosureUnlockReported",
			"func b5(s *stats) {\n func() {\n  s.mu.Lock()\n  goto E\n A:\n  _ = s.total\n  return\n  s.mu.Lock()\n B:\n  goto A\n  s.mu.Lock()\n C:\n  goto B\n  s.mu.Lock()\n D:\n  goto C\n  s.mu.Lock()\n E:\n  s.mu.Unlock()\n  goto D\n }()\n}\n",
			[]string{"34 concurrency::unguarded_access forbid"},
		},
		{
			"fiveLabelClosureRelockedStaysClean",
			"func b6(s *stats) {\n func() {\n  s.mu.Lock()\n  goto E\n A:\n  _ = s.total\n  return\n  s.mu.Lock()\n B:\n  s.mu.Lock()\n  goto A\n  s.mu.Lock()\n C:\n  s.mu.Lock()\n  goto B\n  s.mu.Lock()\n D:\n  s.mu.Lock()\n  goto C\n  s.mu.Lock()\n E:\n  s.mu.Unlock()\n  s.mu.Lock()\n  goto D\n }()\n}\n",
			nil,
		},
		{
			"unlockDeferredInEarlyClosureReports",
			"func m1(s *stats) {\n s.mu.Lock()\n func() {\n  defer s.mu.Unlock()\n }()\n _ = s.total\n}\n",
			[]string{"34 concurrency::unguarded_access forbid"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name+"/lgo", func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", advLocksHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
		t.Run(tt.name+"/go", func(t *testing.T) {
			want := make([]string, 0, len(tt.want))
			for _, w := range tt.want {
				goW, _ := strings.CutSuffix(w, " forbid")
				want = append(want, goW+" deny")
			}
			got := summarize(Run(checkSrc(t, KindGo, "x.go", advLocksHead+tt.body)))
			if !slices.Equal(got, want) {
				t.Errorf("results = %q, want %q", got, want)
			}
		})
	}
}

func TestAdvGuardedAccessLimitations(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"storedClosureReported",
			"func h2(s *stats) int {\n s.mu.Lock()\n defer s.mu.Unlock()\n f := func() int { return s.total }\n return f()\n}\n",
			[]string{"32 concurrency::unguarded_access deny"},
		},
		{
			"backwardGotoInClosureReported",
			"func b3(s *stats) {\n s.mu.Lock()\n func() {\nL:\n  _ = s.total\n  s.mu.Unlock()\n  goto L\n }()\n}\n",
			[]string{"33 concurrency::unguarded_access deny"},
		},
		{
			"backwardGotoInClosureRelockedStaysClean",
			"func b4(s *stats) {\n s.mu.Lock()\n func() {\nL:\n  _ = s.total\n  s.mu.Unlock()\n  s.mu.Lock()\n  goto L\n }()\n}\n",
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindGo, "x.go", advLocksHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

const advRequireHead = `package p

import "sync"

type gate struct {
	mu sync.Mutex

	//logos:guardedby mu
	open bool
}

//logos:requires g.mu
func (g *gate) toggle() {
	g.open = !g.open
}

//logos:releases g.mu
func (g *gate) release() {}
`

func TestAdvLockRequirement(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want []string
	}{
		{
			"backwardGotoDeferredRequireReportsOnce",
			"func rd(g *gate, again bool) {\n defer g.toggle()\nL:\n if again {\n  goto L\n }\n}\n",
			[]string{"20 concurrency::lock_requirement_unmet forbid"},
		},
		{
			"deferredReleaseThenRequire",
			"func r3(g *gate) {\n g.mu.Lock()\n defer g.toggle()\n defer g.release()\n}\n",
			[]string{"21 concurrency::lock_requirement_unmet forbid"},
		},
		{
			"deferredUnlockThenRequire",
			"func m(g *gate) {\n g.mu.Lock()\n defer g.toggle()\n defer g.mu.Unlock()\n}\n",
			[]string{"21 concurrency::lock_requirement_unmet forbid"},
		},
		{
			"directRequireHeld",
			"func r4(g *gate) {\n g.mu.Lock()\n g.toggle()\n g.mu.Unlock()\n}\n",
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", advRequireHead+tt.body)))
			if !slices.Equal(got, tt.want) {
				t.Errorf("results = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAdvLambdaArgReadsGuarded(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const head = `package p

import "sync"

type stats struct {
	mu sync.Mutex

	//logos:guardedby mu
	total int
}

func apply2(f func(int) int) int { return f(1) }
`
	body := "func hl(s *stats) int {\n s.mu.Lock()\n defer s.mu.Unlock()\n return apply2(v => v + s.total)\n}\n"
	got := summarize(Run(checkSrc(t, KindLgo, "x.lgo", head+body)))
	want := []string{"16 concurrency::unguarded_access forbid"}
	if !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}
