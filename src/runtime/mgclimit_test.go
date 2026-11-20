// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
	"time"
)

func TestGCCPULimiter(t *testing.T) {
	const procs = 14

	// Create mock time.
	ticks := int64(0)
	advance := func(d time.Duration) int64 {
		t.Helper()
		ticks += int64(d)
		return ticks
	}

	// assistTime computes the CPU time for assists using frac of GOMAXPROCS
	// over the wall-clock duration d.
	assistTime := func(d time.Duration, frac float64) int64 {
		t.Helper()
		return int64(frac * float64(d) * procs)
	}

	l := NewGCCPULimiter(ticks, procs)

	// Do the whole test twice to make sure state doesn't leak across.
	var baseOverflow uint64 // Track total overflow across iterations.
	for i := 0; i < 2; i++ {
		t.Logf("Iteration %d", i+1)

		if l.Capacity() != procs*CapacityPerProc {
			t.Fatalf("unexpected capacity: %d", l.Capacity())
		}
		if l.Fill() != 0 {
			t.Fatalf("expected empty bucket to start")
		}

		// Test filling the bucket with just mutator time.

		l.Update(advance(10 * time.Millisecond))
		l.Update(advance(1 * time.Second))
		l.Update(advance(1 * time.Hour))
		if l.Fill() != 0 {
			t.Fatalf("expected empty bucket from only accumulating mutator time, got fill of %d cpu-ns", l.Fill())
		}

		// Test needUpdate.

		if l.NeedUpdate(advance(GCCPULimiterUpdatePeriod / 2)) {
			t.Fatal("need update even though updated half a period ago")
		}
		if !l.NeedUpdate(advance(GCCPULimiterUpdatePeriod)) {
			t.Fatal("doesn't need update even though updated 1.5 periods ago")
		}
		l.Update(advance(0))
		if l.NeedUpdate(advance(0)) {
			t.Fatal("need update even though just updated")
		}

		// Test transitioning the bucket to enable the GC.

		l.StartGCTransition(true, advance(109*time.Millisecond))
		l.FinishGCTransition(advance(2*time.Millisecond + 1*time.Microsecond))

		if expect := uint64((2*time.Millisecond + 1*time.Microsecond) * procs); l.Fill() != expect {
			t.Fatalf("expected fill of %d, got %d cpu-ns", expect, l.Fill())
		}

		// Test passing time without assists during a GC. Specifically, just enough to drain the bucket to
		// exactly procs nanoseconds (easier to get to because of rounding).
		//
		// The window we need to drain the bucket is 1/(1-2*gcBackgroundUtilization) times the current fill:
		//
		//   fill + (window * procs * gcBackgroundUtilization - window * procs * (1-gcBackgroundUtilization)) = n
		//   fill = n - (window * procs * gcBackgroundUtilization - window * procs * (1-gcBackgroundUtilization))
		//   fill = n + window * procs * ((1-gcBackgroundUtilization) - gcBackgroundUtilization)
		//   fill = n + window * procs * (1-2*gcBackgroundUtilization)
		//   window = (fill - n) / (procs * (1-2*gcBackgroundUtilization)))
		//
		// And here we want n=procs:
		factor := (1 / (1 - 2*GCBackgroundUtilization))
		fill := (2*time.Millisecond + 1*time.Microsecond) * procs
		l.Update(advance(time.Duration(factor * float64(fill-procs) / procs)))
		if l.Fill() != procs {
			t.Fatalf("expected fill %d cpu-ns from draining after a GC started, got fill of %d cpu-ns", procs, l.Fill())
		}

		// Drain to zero for the rest of the test.
		l.Update(advance(2 * procs * CapacityPerProc))
		if l.Fill() != 0 {
			t.Fatalf("expected empty bucket from draining, got fill of %d cpu-ns", l.Fill())
		}

		// Test filling up the bucket with 50% total GC work (so, not moving the bucket at all).
		l.AddAssistTime(assistTime(10*time.Millisecond, 0.5-GCBackgroundUtilization))
		l.Update(advance(10 * time.Millisecond))
		if l.Fill() != 0 {
			t.Fatalf("expected empty bucket from 50%% GC work, got fill of %d cpu-ns", l.Fill())
		}

		// Test adding to the bucket overall with 100% GC work.
		l.AddAssistTime(assistTime(time.Millisecond, 1.0-GCBackgroundUtilization))
		l.Update(advance(time.Millisecond))
		if expect := uint64(procs * time.Millisecond); l.Fill() != expect {
			t.Errorf("expected %d fill from 100%% GC CPU, got fill of %d cpu-ns", expect, l.Fill())
		}
		if l.Limiting() {
			t.Errorf("limiter is enabled after filling bucket but shouldn't be")
		}
		if t.Failed() {
			t.FailNow()
		}

		// Test filling the bucket exactly full.
		l.AddAssistTime(assistTime(CapacityPerProc-time.Millisecond, 1.0-GCBackgroundUtilization))
		l.Update(advance(CapacityPerProc - time.Millisecond))
		if l.Fill() != l.Capacity() {
			t.Errorf("expected bucket filled to capacity %d, got %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is not enabled after filling bucket but should be")
		}
		if l.Overflow() != 0+baseOverflow {
			t.Errorf("bucket filled exactly should not have overflow, found %d", l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Test adding with a delta of exactly zero. That is, GC work is exactly 50% of all resources.
		// Specifically, the limiter should still be on, and no overflow should accumulate.
		l.AddAssistTime(assistTime(1*time.Second, 0.5-GCBackgroundUtilization))
		l.Update(advance(1 * time.Second))
		if l.Fill() != l.Capacity() {
			t.Errorf("expected bucket filled to capacity %d, got %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is not enabled after filling bucket but should be")
		}
		if l.Overflow() != 0+baseOverflow {
			t.Errorf("bucket filled exactly should not have overflow, found %d", l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Drain the bucket by half.
		l.AddAssistTime(assistTime(CapacityPerProc, 0))
		l.Update(advance(CapacityPerProc))
		//
		// test start condition: l.Fill == l.Capacity
		//
		// assistTime   = 0
		// gcTime       = CapacityPerProc * procs * GCBackgroundUtilization
		// mutatorTime  = CapacityPerProc * procs * (1 - GCBackgroundUtilization)
		// change       = gcTime - mutatorTime
		//              = CapacityPerProc * procs * (2 * GCBackgroundUtilization - 1)
		// l.Fill       = l.Fill + change       // here change < 0
		//              = l.Capacity + change
		//              = CapacityPerProc * procs + change
		//              = CapacityPerProc * procs * 2 * GCBackgroundUtilization
		//              = l.Capacity * 2 * GCBackgroundUtilization
		//              = l.Capacity * 0.5      // by default GCBackgroundUtilization = 0.25
		//              = l.Capacity / 2
		//
		if expect := l.Capacity() / 2; l.Fill() != expect {
			t.Errorf("failed to drain to %d, got fill %d", expect, l.Fill())
		}
		if l.Limiting() {
			t.Errorf("limiter is enabled after draining bucket but shouldn't be")
		}
		if t.Failed() {
			t.FailNow()
		}

		// Test overfilling the bucket.
		l.AddAssistTime(assistTime(CapacityPerProc, 1.0-GCBackgroundUtilization))
		l.Update(advance(CapacityPerProc))
		//
		// test start condition: l.Fill == l.Capacity * 2 * GCBackgroundUtilization
		//
		// assistTime   = CapacityPerProc * procs * (1 - GCBackgroundUtilization)
		// gcTime       = CapacityPerProc * procs * GCBackgroundUtilization + assistTime
		//              = CapacityPerProc * procs
		//              = l.Capacity
		// mutatorTime  = CapacityPerProc * procs * (1 - GCBackgroundUtilization) - assistTime
		//              = 0
		// change       = gcTime - mutatorTime
		//              = l.Capacity
		// l.Overflow   = l.Fill + change - l.Capacity
		//              = l.Capacity * 2 * GCBackgroundUtilization + l.Capacity - l.Capacity
		//              = l.Capacity * 2 * GCBackgroundUtilization
		//              = l.Capacity * 0.5  // by default GCBackgroundUtilization = 0.25
		//              = l.Capacity / 2
		//              = CapacityPerProc * procs / 2
		// l.Fill       = l.Capacity        // because change is too large
		//
		if l.Fill() != l.Capacity() {
			t.Errorf("failed to fill to capacity %d, got fill %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is not enabled after overfill but should be")
		}
		if expect := uint64(CapacityPerProc * procs / 2); l.Overflow() != expect+baseOverflow {
			t.Errorf("bucket overfilled should have overflow %d, found %d", expect, l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Test ending the cycle with some assists left over.
		l.AddAssistTime(assistTime(1*time.Millisecond, 1.0-GCBackgroundUtilization))
		l.StartGCTransition(false, advance(1*time.Millisecond))
		if l.Fill() != l.Capacity() {
			t.Errorf("failed to maintain fill to capacity %d, got fill %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is not enabled after overfill but should be")
		}
		if expect := uint64((CapacityPerProc/2 + time.Millisecond) * procs); l.Overflow() != expect+baseOverflow {
			t.Errorf("bucket overfilled should have overflow %d, found %d", expect, l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Make sure the STW adds to the bucket.
		l.FinishGCTransition(advance(5 * time.Millisecond))
		if l.Fill() != l.Capacity() {
			t.Errorf("failed to maintain fill to capacity %d, got fill %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is not enabled after overfill but should be")
		}
		if expect := uint64((CapacityPerProc/2 + 6*time.Millisecond) * procs); l.Overflow() != expect+baseOverflow {
			t.Errorf("bucket overfilled should have overflow %d, found %d", expect, l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Resize procs up and make sure limiting stops.
		expectFill := l.Capacity()
		l.ResetCapacity(advance(0), procs+10)
		if l.Fill() != expectFill {
			t.Errorf("failed to maintain fill at old capacity %d, got fill %d", expectFill, l.Fill())
		}
		if l.Limiting() {
			t.Errorf("limiter is enabled after resetting capacity higher")
		}
		if expect := uint64((CapacityPerProc/2 + 6*time.Millisecond) * procs); l.Overflow() != expect+baseOverflow {
			t.Errorf("bucket overflow %d should have remained constant, found %d", expect, l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Resize procs down and make sure limiting begins again.
		// Also make sure resizing doesn't affect overflow. This isn't
		// a case where we want to report overflow, because we're not
		// actively doing work to achieve it. It's that we have fewer
		// CPU resources now.
		l.ResetCapacity(advance(0), procs-10)
		if l.Fill() != l.Capacity() {
			t.Errorf("failed lower fill to new capacity %d, got fill %d", l.Capacity(), l.Fill())
		}
		if !l.Limiting() {
			t.Errorf("limiter is disabled after resetting capacity lower")
		}
		if expect := uint64((CapacityPerProc/2 + 6*time.Millisecond) * procs); l.Overflow() != expect+baseOverflow {
			t.Errorf("bucket overflow %d should have remained constant, found %d", expect, l.Overflow())
		}
		if t.Failed() {
			t.FailNow()
		}

		// Get back to a zero state. The top of the loop will double check.
		l.ResetCapacity(advance(CapacityPerProc*procs), procs)

		// Track total overflow for future iterations.
		baseOverflow += uint64((CapacityPerProc/2 + 6*time.Millisecond) * procs)
	}
}
