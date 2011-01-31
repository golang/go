// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	. "time"
)

func TestTicker(t *testing.T) {
	const (
		Delta = 100 * 1e6
		Count = 10
	)
	ticker := NewTicker(Delta)
	t0 := Nanoseconds()
	for i := 0; i < Count; i++ {
		<-ticker.C
	}
	ticker.Stop()
	t1 := Nanoseconds()
	ns := t1 - t0
	target := int64(Delta * Count)
	slop := target * 2 / 10
	if ns < target-slop || ns > target+slop {
		t.Fatalf("%d ticks of %g ns took %g ns, expected %g", Count, float64(Delta), float64(ns), float64(target))
	}
	// Now test that the ticker stopped
	Sleep(2 * Delta)
	select {
	case <-ticker.C:
		t.Fatal("Ticker did not shut down")
	default:
		// ok
	}
}

// Test that a bug tearing down a ticker has been fixed.  This routine should not deadlock.
func TestTeardown(t *testing.T) {
	for i := 0; i < 3; i++ {
		ticker := NewTicker(1e8)
		<-ticker.C
		ticker.Stop()
	}
}

func BenchmarkTicker(b *testing.B) {
	ticker := NewTicker(1)
	b.ResetTimer()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		<-ticker.C
	}
	b.StopTimer()
	ticker.Stop()
}
