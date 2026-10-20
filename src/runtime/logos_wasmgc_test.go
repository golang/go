// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math/rand/v2"
	"runtime"
	"testing"
)

// The wasm collector marks in slices during idle time of the JS event
// loop (SPEC Sec 19, T-301.2). A slice runs for at most
// LogosWasmSliceNanos; after a slice ends at its deadline the idle path
// returns to the event loop before the next slice starts. These tests pin
// that state machine with an explicit clock.

// A slice ends exactly at its deadline, once, and only then does it park
// the idle path.
func TestLogosWasmSliceEndsAtDeadline(t *testing.T) {
	var s runtime.LogosWasmSlicer
	if !s.MayStart() {
		t.Fatal("fresh slicer refuses to start a slice")
	}
	const t0 = 1_000_000
	s.Begin(t0, 0)
	if s.Expired(t0 + runtime.LogosWasmSliceNanos - 1) {
		t.Fatal("slice expired one nanosecond before its deadline")
	}
	if s.Parked() || !s.MayStart() {
		t.Fatal("an unexpired slice must not park the idle path")
	}
	if !s.Expired(t0 + runtime.LogosWasmSliceNanos) {
		t.Fatal("slice did not expire at its deadline")
	}
	if s.Expired(t0 + 2*runtime.LogosWasmSliceNanos) {
		t.Fatal("one slice expired twice")
	}
	if !s.Parked() || s.MayStart() {
		t.Fatal("an expired slice must park the idle path until an event turn")
	}
}

// Only an event loop turn lets the next slice start.
func TestLogosWasmNextSliceWaitsForEventTurn(t *testing.T) {
	var s runtime.LogosWasmSlicer
	s.Begin(0, 0)
	s.Expired(runtime.LogosWasmSliceNanos)
	for i := range 3 {
		if s.MayStart() {
			t.Fatalf("slice allowed before an event turn (check %d)", i)
		}
	}
	s.EventTurn()
	if !s.MayStart() || s.Parked() {
		t.Fatal("event turn did not release the idle path")
	}
	s.EventTurn()
	if !s.MayStart() {
		t.Fatal("a second event turn must be harmless")
	}
}

// A slice that stops before its deadline (work ran out, or another
// goroutine became runnable) never parks the idle path.
func TestLogosWasmEarlyStopDoesNotPark(t *testing.T) {
	var s runtime.LogosWasmSlicer
	s.Begin(0, 0)
	s.Begin(10, 0)
	if s.Parked() || !s.MayStart() {
		t.Fatal("a slice restarted before its deadline parked the idle path")
	}
}

// The idle path asks the event loop to call back soon while a parked
// slice leaves mark work behind, and leaves an earlier Go timer alone.
func TestLogosWasmWakeAt(t *testing.T) {
	var s runtime.LogosWasmSlicer
	const now = 5_000_000
	if got := s.WakeAt(now, 0); got != 0 {
		t.Fatalf("unparked WakeAt(now, 0) = %d, want 0", got)
	}
	if got := s.WakeAt(now, now+7); got != now+7 {
		t.Fatalf("unparked WakeAt keeps pollUntil: got %d, want %d", got, now+7)
	}
	s.Begin(0, 0)
	s.Expired(runtime.LogosWasmSliceNanos)
	if got, want := s.WakeAt(now, 0), int64(now+runtime.LogosWasmWakeNanos); got != want {
		t.Fatalf("parked WakeAt(now, 0) = %d, want %d", got, want)
	}
	later := int64(now + 10*runtime.LogosWasmWakeNanos)
	if got, want := s.WakeAt(now, later), int64(now+runtime.LogosWasmWakeNanos); got != want {
		t.Fatalf("parked WakeAt with a later timer = %d, want %d", got, want)
	}
	earlier := int64(now + runtime.LogosWasmWakeNanos/2)
	if got := s.WakeAt(now, earlier); got != earlier {
		t.Fatalf("parked WakeAt with an earlier timer = %d, want %d", got, earlier)
	}
}

// Every slice records the id of the M it ran on; the ring keeps the most
// recent LogosWasmSliceRing ids, and the count keeps going past it.
func TestLogosWasmSliceRecordsM(t *testing.T) {
	var s runtime.LogosWasmSlicer
	n := runtime.LogosWasmSliceRing + 5
	for i := range n {
		s.Begin(int64(i), int64(100+i))
	}
	mids := make([]int64, runtime.LogosWasmSliceRing)
	slices, _ := s.Stats(mids)
	if slices != uint64(n) {
		t.Fatalf("slices = %d, want %d", slices, n)
	}
	seen := make(map[int64]bool)
	for _, m := range mids {
		seen[m] = true
	}
	for i := n - runtime.LogosWasmSliceRing; i < n; i++ {
		if !seen[int64(100+i)] {
			t.Fatalf("m id %d of slice %d missing from the ring", 100+i, i)
		}
	}
	short := make([]int64, 2)
	if got, _ := s.Stats(short); got != uint64(n) {
		t.Fatalf("Stats with a short buffer changed the count to %d", got)
	}
}

// The longest slice that ended at its deadline is the maximum of the
// observed lengths, and a slice that ended early does not lower it.
func TestLogosWasmLongestSlice(t *testing.T) {
	var s runtime.LogosWasmSlicer
	s.Begin(0, 0)
	s.Expired(runtime.LogosWasmSliceNanos + 700)
	s.EventTurn()
	s.Begin(10_000_000, 0)
	s.Expired(10_000_000 + runtime.LogosWasmSliceNanos + 50)
	_, longest := s.Stats(nil)
	if want := int64(runtime.LogosWasmSliceNanos + 700); longest != want {
		t.Fatalf("longest = %d, want %d", longest, want)
	}
	if _, again := s.Stats(nil); again != 0 {
		t.Fatalf("longest = %d on the next read, want 0: the window restarts at each read", again)
	}
}

// The clock is checked at the credit slack under slices: a threshold
// above it is clamped, one at or below it is left alone.
func TestLogosWasmCheckWorkClamp(t *testing.T) {
	const upstream = 100000
	for _, c := range []struct{ in, want int64 }{
		{upstream, runtime.LogosWasmCheckWork},
		{runtime.LogosWasmCheckWork + 1, runtime.LogosWasmCheckWork},
		{runtime.LogosWasmCheckWork, runtime.LogosWasmCheckWork},
		{runtime.LogosWasmCheckWork - 1, runtime.LogosWasmCheckWork - 1},
		{1, 1},
	} {
		if got := runtime.LogosClampCheckWork(c.in); got != c.want {
			t.Errorf("LogosClampCheckWork(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

// Without mark slices in force (every target but js/wasm linked with
// -gc=wasm) the scheduler hooks keep the upstream behavior: the drain
// threshold is unchanged and the mark workers are never gated.
func TestLogosWasmSlicesOffKeepUpstream(t *testing.T) {
	if runtime.LogosWasmSliceActive() {
		t.Skip("mark slices are in force on this binary")
	}
	const upstream = 100000
	if got := runtime.LogosDrainCheckThreshold(upstream); got != upstream {
		t.Fatalf("LogosDrainCheckThreshold(%d) = %d without slices", upstream, got)
	}
	if !runtime.LogosWasmWorkersMayRun() {
		t.Fatal("mark workers gated without slices")
	}
}

// A dedicated worker drains without a deadline and wasm has no preemption
// to interrupt it, so mark slices start every cycle without one.
func TestLogosWasmNoDedicatedWorkers(t *testing.T) {
	for _, needed := range []int64{0, 1, 2, 4} {
		want := needed
		if runtime.LogosWasmSliceActive() {
			want = 0
		}
		if got := runtime.LogosWasmDedicatedWorkers(needed); got != want {
			t.Errorf("LogosWasmDedicatedWorkers(%d) = %d, want %d", needed, got, want)
		}
	}
}

// A random walk over the slice events agrees with an independent model:
// the slicer is parked exactly from an expiry to the next event turn, a
// parked slicer refuses slices, and one slice expires at most once.
func TestLogosWasmSlicerInvariants(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 19))
	var s runtime.LogosWasmSlicer
	now := int64(1)
	running := false
	parked := false
	for i := range 20000 {
		now += int64(rng.IntN(runtime.LogosWasmSliceNanos))
		switch rng.IntN(4) {
		case 0:
			if s.MayStart() {
				s.Begin(now, 0)
				running = true
			}
		case 1:
			if s.Expired(now) {
				if !running {
					t.Fatalf("step %d: expiry without a running slice", i)
				}
				running = false
				parked = true
			}
		case 2:
			s.EventTurn()
			parked = false
		case 3:
			s.WakeAt(now, int64(rng.IntN(3))*now)
		}
		if s.Parked() != parked {
			t.Fatalf("step %d: Parked() = %v, model says %v", i, s.Parked(), parked)
		}
		if s.MayStart() == parked {
			t.Fatalf("step %d: MayStart() = %v while parked = %v", i, s.MayStart(), parked)
		}
	}
}
