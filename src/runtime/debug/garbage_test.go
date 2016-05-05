// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

import (
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
			t.Errorf("stats.PauseEnd[%d] = %d, want %d", i, dt, mstats.PauseEnd[off])
		}
		off = (off + len(mstats.PauseEnd) - 1) % len(mstats.PauseEnd)
	}
}

var big = make([]byte, 1<<20)

func TestFreeOSMemory(t *testing.T) {
	if runtime.GOARCH == "arm64" || runtime.GOARCH == "ppc64" || runtime.GOARCH == "ppc64le" || runtime.GOARCH == "mips64" || runtime.GOARCH == "mips64le" ||
		runtime.GOOS == "nacl" {
		t.Skip("issue 9993; scavenger temporarily disabled on systems with physical pages larger than logical pages")
	}
	var ms1, ms2 runtime.MemStats

	if big == nil {
		t.Skip("test is not reliable when run multiple times")
	}
	big = nil
	runtime.GC()
	runtime.ReadMemStats(&ms1)
	FreeOSMemory()
	runtime.ReadMemStats(&ms2)
	if ms1.HeapReleased >= ms2.HeapReleased {
		t.Errorf("released before=%d; released after=%d; did not go up", ms1.HeapReleased, ms2.HeapReleased)
	}
}

func TestSetGCPercent(t *testing.T) {
	// Test that the variable is being set and returned correctly.
	// Assume the percentage itself is implemented fine during GC,
	// which is harder to test.
	old := SetGCPercent(123)
	new := SetGCPercent(old)
	if new != 123 {
		t.Errorf("SetGCPercent(123); SetGCPercent(x) = %d, want 123", new)
	}
}
