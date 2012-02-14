// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	. "time"
)

func TestTicker(t *testing.T) {
	const Count = 10
	Delta := 100 * Millisecond
	if testing.Short() {
		Delta = 10 * Millisecond
	}
	ticker := NewTicker(Delta)
	t0 := Now()
	for i := 0; i < Count; i++ {
		<-ticker.C
	}
	ticker.Stop()
	t1 := Now()
	dt := t1.Sub(t0)
	target := Delta * Count
	slop := target * 2 / 10
	if dt < target-slop || dt > target+slop {
		t.Fatalf("%d %s ticks took %s, expected [%s,%s]", Count, Delta, dt, target-slop, target+slop)
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
	Delta := 100 * Millisecond
	if testing.Short() {
		Delta = 20 * Millisecond
	}
	for i := 0; i < 3; i++ {
		ticker := NewTicker(Delta)
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
