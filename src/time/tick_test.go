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
	if dt < target-slop || (!testing.Short() && dt > target+slop) {
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

// Test that a bug tearing down a ticker has been fixed. This routine should not deadlock.
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

// Test the Tick convenience wrapper.
func TestTick(t *testing.T) {
	// Test that giving a negative duration returns nil.
	if got := Tick(-1); got != nil {
		t.Errorf("Tick(-1) = %v; want nil", got)
	}
}

// Test that NewTicker panics when given a duration less than zero.
func TestNewTickerLtZeroDuration(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Errorf("NewTicker(-1) should have panicked")
		}
	}()
	NewTicker(-1)
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
