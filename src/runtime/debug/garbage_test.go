// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
	"internal/testenv"
	"os"
	"runtime"
	. "runtime/debug"
	"testing"
	"time"
)

func TestReadGCStats(t *testing.T) {
	defer SetGCPercent(SetGCPercent(-1))

	var stats GCStats
	var mstats runtime.MemStats
	var min, max time.Duration

	// First ReadGCStats will allocate, second should not,
	// especially if we follow up with an explicit garbage collection.
	stats.PauseQuantiles = make([]time.Duration, 10)
	ReadGCStats(&stats)
	runtime.GC()

	// Assume these will return same data: no GC during ReadGCStats.
	ReadGCStats(&stats)
	runtime.ReadMemStats(&mstats)

	if stats.NumGC != int64(mstats.NumGC) {
		t.Errorf("stats.NumGC = %d, but mstats.NumGC = %d", stats.NumGC, mstats.NumGC)
	}
	if stats.PauseTotal != time.Duration(mstats.PauseTotalNs) {
		t.Errorf("stats.PauseTotal = %d, but mstats.PauseTotalNs = %d", stats.PauseTotal, mstats.PauseTotalNs)
	}
	if stats.LastGC.UnixNano() != int64(mstats.LastGC) {
		t.Errorf("stats.LastGC.UnixNano = %d, but mstats.LastGC = %d", stats.LastGC.UnixNano(), mstats.LastGC)
	}
	n := int(mstats.NumGC)
	if n > len(mstats.PauseNs) {
		n = len(mstats.PauseNs)
	}
	if len(stats.Pause) != n {
		t.Errorf("len(stats.Pause) = %d, want %d", len(stats.Pause), n)
	} else {
		off := (int(mstats.NumGC) + len(mstats.PauseNs) - 1) % len(mstats.PauseNs)
		for i := 0; i < n; i++ {
			dt := stats.Pause[i]
			if dt != time.Duration(mstats.PauseNs[off]) {
				t.Errorf("stats.Pause[%d] = %d, want %d", i, dt, mstats.PauseNs[off])
			}
			if max < dt {
				max = dt
			}
			if min > dt || i == 0 {
				min = dt
			}
			off = (off + len(mstats.PauseNs) - 1) % len(mstats.PauseNs)
		}
	}

	q := stats.PauseQuantiles
	nq := len(q)
	if q[0] != min || q[nq-1] != max {
		t.Errorf("stats.PauseQuantiles = [%d, ..., %d], want [%d, ..., %d]", q[0], q[nq-1], min, max)
	}

	for i := 0; i < nq-1; i++ {
		if q[i] > q[i+1] {
			t.Errorf("stats.PauseQuantiles[%d]=%d > stats.PauseQuantiles[%d]=%d", i, q[i], i+1, q[i+1])
		}
	}

	// compare memory stats with gc stats:
	if len(stats.PauseEnd) != n {
		t.Fatalf("len(stats.PauseEnd) = %d, want %d", len(stats.PauseEnd), n)
	}
	off := (int(mstats.NumGC) + len(mstats.PauseEnd) - 1) % len(mstats.PauseEnd)
	for i := 0; i < n; i++ {
		dt := stats.PauseEnd[i]
		if dt.UnixNano() != int64(mstats.PauseEnd[off]) {
			t.Errorf("stats.PauseEnd[%d] = %d, want %d", i, dt.UnixNano(), mstats.PauseEnd[off])
		}
		off = (off + len(mstats.PauseEnd) - 1) % len(mstats.PauseEnd)
	}
}

var big []byte

func TestFreeOSMemory(t *testing.T) {
	// Tests FreeOSMemory by making big susceptible to collection
	// and checking that at least that much memory is returned to
	// the OS after.

	const bigBytes = 32 << 20
	big = make([]byte, bigBytes)

	// Make sure any in-progress GCs are complete.
	runtime.GC()

	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	// Clear the last reference to the big allocation, making it
	// susceptible to collection.
	big = nil

	// FreeOSMemory runs a GC cycle before releasing memory,
	// so it's fine to skip a GC here.
	//
	// It's possible the background scavenger runs concurrently
	// with this function and does most of the work for it.
	// If that happens, it's OK. What we want is a test that fails
	// often if FreeOSMemory does not work correctly, and a test
	// that passes every time if it does.
	FreeOSMemory()

	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	// Check to make sure that the big allocation (now freed)
	// had its memory shift into HeapReleased as a result of that
	// FreeOSMemory.
	if after.HeapReleased <= before.HeapReleased {
		t.Fatalf("no memory released: %d -> %d", before.HeapReleased, after.HeapReleased)
	}

	// Check to make sure bigBytes was released, plus some slack. Pages may get
	// allocated in between the two measurements above for a variety for reasons,
	// most commonly for GC work bufs. Since this can get fairly high, depending
	// on scheduling and what GOMAXPROCS is, give a lot of slack up-front.
	//
	// Add a little more slack too if the page size is bigger than the runtime page size.
	// "big" could end up unaligned on its ends, forcing the scavenger to skip at worst
	// 2x pages.
	slack := uint64(bigBytes / 2)
	pageSize := uint64(os.Getpagesize())
	if pageSize > 8<<10 {
		slack += pageSize * 2
	}
	if slack > bigBytes {
		// We basically already checked this.
		return
	}
	if after.HeapReleased-before.HeapReleased < bigBytes-slack {
		t.Fatalf("less than %d released: %d -> %d", bigBytes, before.HeapReleased, after.HeapReleased)
	}
}

var (
	setGCPercentBallast any
	setGCPercentSink    any
)

func TestSetGCPercent(t *testing.T) {
	testenv.SkipFlaky(t, 20076)

	// Test that the variable is being set and returned correctly.
	old := SetGCPercent(123)
	new := SetGCPercent(old)
	if new != 123 {
		t.Errorf("SetGCPercent(123); SetGCPercent(x) = %d, want 123", new)
	}

	// Test that the percentage is implemented correctly.
	defer func() {
		SetGCPercent(old)
		setGCPercentBallast, setGCPercentSink = nil, nil
	}()
	SetGCPercent(100)
	runtime.GC()
	// Create 100 MB of live heap as a baseline.
	const baseline = 100 << 20
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	setGCPercentBallast = make([]byte, baseline-ms.Alloc)
	runtime.GC()
	runtime.ReadMemStats(&ms)
	if abs64(baseline-int64(ms.Alloc)) > 10<<20 {
		t.Fatalf("failed to set up baseline live heap; got %d MB, want %d MB", ms.Alloc>>20, baseline>>20)
	}
	// NextGC should be ~200 MB.
	const thresh = 20 << 20 // TODO: Figure out why this is so noisy on some builders
	if want := int64(2 * baseline); abs64(want-int64(ms.NextGC)) > thresh {
		t.Errorf("NextGC = %d MB, want %d±%d MB", ms.NextGC>>20, want>>20, thresh>>20)
	}
	// Create some garbage, but not enough to trigger another GC.
	for i := 0; i < int(1.2*baseline); i += 1 << 10 {
		setGCPercentSink = make([]byte, 1<<10)
	}
	setGCPercentSink = nil
	// Adjust GOGC to 50. NextGC should be ~150 MB.
	SetGCPercent(50)
	runtime.ReadMemStats(&ms)
	if want := int64(1.5 * baseline); abs64(want-int64(ms.NextGC)) > thresh {
		t.Errorf("NextGC = %d MB, want %d±%d MB", ms.NextGC>>20, want>>20, thresh>>20)
	}

	// Trigger a GC and get back to 100 MB live with GOGC=100.
	SetGCPercent(100)
	runtime.GC()
	// Raise live to 120 MB.
	setGCPercentSink = make([]byte, int(0.2*baseline))
	// Lower GOGC to 10. This must force a GC.
	runtime.ReadMemStats(&ms)
	ngc1 := ms.NumGC
	SetGCPercent(10)
	// It may require an allocation to actually force the GC.
	setGCPercentSink = make([]byte, 1<<20)
	runtime.ReadMemStats(&ms)
	ngc2 := ms.NumGC
	if ngc1 == ngc2 {
		t.Errorf("expected GC to run but it did not")
	}
}

func abs64(a int64) int64 {
	if a < 0 {
		return -a
	}
	return a
}

func TestSetMaxThreadsOvf(t *testing.T) {
	// Verify that a big threads count will not overflow the int32
	// maxmcount variable, causing a panic (see Issue 16076).
	//
	// This can only happen when ints are 64 bits, since on platforms
	// with 32 bit ints SetMaxThreads (which takes an int parameter)
	// cannot be given anything that will overflow an int32.
	//
	// Call SetMaxThreads with 1<<31, but only on 64 bit systems.
	nt := SetMaxThreads(1 << (30 + ^uint(0)>>63))
	SetMaxThreads(nt) // restore previous value
}
