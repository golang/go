// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Attack tests for M1-S4: direct TraceErr/ErrorTrace behavior without ?
// syntax. The three-function ? propagation chain is owned by M1-S5 and
// is not covered here. Signatures follow the M1-S4 plan text verbatim:
// type Site struct{ File string; Line int },
// func TraceErr(err error, s *Site), func ErrorTrace() []Site.
// Calls use bare-statement form so they also compile if TraceErr
// returns its error argument. Nil-error behavior is unspecified by the
// plan and SPEC and is deliberately not exercised here. Each recorded
// Site uses a distinct variable that outlives its read, so the file
// passes whether the ring copies values or retains pointers.
package rt_test

import (
	"errors"
	"runtime"
	"slices"
	"sync"
	"testing"

	"logos/rt"
)

func checkTraceSites(t *testing.T, got, want []rt.Site) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("ErrorTrace length = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i].File != want[i].File || got[i].Line != want[i].Line {
			t.Errorf("ErrorTrace()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestLogosTraceThreeSitesInOrder(t *testing.T) {
	err := errors.New("order")
	s1 := rt.Site{File: "order_a.go", Line: 11}
	s2 := rt.Site{File: "order_b.go", Line: 22}
	s3 := rt.Site{File: "order_c.go", Line: 33}
	rt.TraceErr(err, &s1)
	rt.TraceErr(err, &s2)
	rt.TraceErr(err, &s3)
	checkTraceSites(t, rt.ErrorTrace(), []rt.Site{s1, s2, s3})
}

func TestLogosTraceDifferentErrorResets(t *testing.T) {
	err1 := errors.New("first")
	err2 := errors.New("second")
	a := rt.Site{File: "reset_a.go", Line: 1}
	b := rt.Site{File: "reset_b.go", Line: 2}
	c := rt.Site{File: "reset_c.go", Line: 3}
	rt.TraceErr(err1, &a)
	rt.TraceErr(err1, &b)
	rt.TraceErr(err2, &c)
	checkTraceSites(t, rt.ErrorTrace(), []rt.Site{c})
}

// TestLogosTraceGoroutineIndependent requires per-goroutine traces:
// two goroutines recording concurrently must not observe each other.
// Intentionally no t.Parallel: isolation is asserted with an explicit
// child goroutine joined by sync.WaitGroup.
func TestLogosTraceGoroutineIndependent(t *testing.T) {
	errMain := errors.New("main goroutine")
	errChild := errors.New("child goroutine")
	m1 := rt.Site{File: "main_a.go", Line: 101}
	m2 := rt.Site{File: "main_b.go", Line: 102}
	b1 := rt.Site{File: "child_a.go", Line: 201}
	b2 := rt.Site{File: "child_b.go", Line: 202}

	type childResult struct {
		initial int
		sites   []rt.Site
	}
	childCh := make(chan childResult, 1)
	var wg sync.WaitGroup
	wg.Go(func() {
		initial := len(rt.ErrorTrace())
		rt.TraceErr(errChild, &b1)
		rt.TraceErr(errChild, &b2)
		cp := slices.Clone(rt.ErrorTrace())
		childCh <- childResult{initial: initial, sites: cp}
	})
	rt.TraceErr(errMain, &m1)
	rt.TraceErr(errMain, &m2)
	wg.Wait()
	res := <-childCh
	if res.initial != 0 {
		t.Errorf("fresh goroutine ErrorTrace length = %d, want 0 (trace leaked across goroutines)", res.initial)
	}
	checkTraceSites(t, res.sites, []rt.Site{b1, b2})
	checkTraceSites(t, rt.ErrorTrace(), []rt.Site{m1, m2})
}

// TestLogosTraceRingKeepsLastEight checks the ring bound at N-1, N, and
// N+1: 7 sites are all kept, 8 sites are all kept, and longer traces
// keep exactly the last 8 in order.
func TestLogosTraceRingKeepsLastEight(t *testing.T) {
	for _, tc := range []struct {
		name string
		n    int
	}{
		{"seven", 7},
		{"eight", 8},
		{"nine", 9},
		{"twelve", 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := errors.New("ring")
			recorded := make([]rt.Site, 0, tc.n)
			for i := range tc.n {
				recorded = append(recorded, rt.Site{File: "ring.go", Line: 1000 + i})
			}
			for i := range recorded {
				rt.TraceErr(err, &recorded[i])
			}
			want := recorded
			if len(recorded) > 8 {
				want = recorded[len(recorded)-8:]
			}
			checkTraceSites(t, rt.ErrorTrace(), want)
		})
	}
}

func TestLogosTraceNoAlloc(t *testing.T) {
	err := errors.New("noalloc")
	s := rt.Site{File: "noalloc.go", Line: 1}
	if n := testing.AllocsPerRun(200, func() {
		rt.TraceErr(err, &s)
	}); n != 0 {
		t.Fatalf("TraceErr allocated %v per run, want 0", n)
	}
}

// TestLogosTraceClearedOnGoroutineExit proves the gdestroy hook clears the
// ring: a goroutine records a trace and exits, and a later goroutine that
// reuses its g must read an empty trace. Deleting the logosClearErrTrace
// call in gdestroy leaves this test red. GOMAXPROCS is pinned to 1 so each
// reader deterministically reuses the writer's just-freed g: the local
// gfree list is LIFO, and with one P the writer runs goexit and gdestroy
// before the waiting test goroutine is rescheduled.
func TestLogosTraceClearedOnGoroutineExit(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	site := rt.Site{File: "exit.go", Line: 7}
	for range 100 {
		err := errors.New("exit")
		var wg sync.WaitGroup
		wg.Go(func() {
			rt.TraceErr(err, &site)
		})
		wg.Wait()
		var got []rt.Site
		wg.Go(func() {
			got = rt.ErrorTrace()
		})
		wg.Wait()
		if len(got) != 0 {
			t.Fatalf("reused goroutine ErrorTrace length = %d, want 0 (stale trace survived goroutine exit: %v)", len(got), got)
		}
	}
}
