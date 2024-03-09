// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	. "time"
)

func TestTicker(t *testing.T) {
	// We want to test that a ticker takes as much time as expected.
	// Since we don't want the test to run for too long, we don't
	// want to use lengthy times. This makes the test inherently flaky.
	// Start with a short time, but try again with a long one if the
	// first test fails.

	baseCount := 10
	baseDelta := 20 * Millisecond

	// On Darwin ARM64 the tick frequency seems limited. Issue 35692.
	if (runtime.GOOS == "darwin" || runtime.GOOS == "ios") && runtime.GOARCH == "arm64" {
		// The following test will run ticker count/2 times then reset
		// the ticker to double the duration for the rest of count/2.
		// Since tick frequency is limited on Darwin ARM64, use even
		// number to give the ticks more time to let the test pass.
		// See CL 220638.
		baseCount = 6
		baseDelta = 100 * Millisecond
	}

	var errs []string
	logErrs := func() {
		for _, e := range errs {
			t.Log(e)
		}
	}

	for _, test := range []struct {
		count int
		delta Duration
	}{{
		count: baseCount,
		delta: baseDelta,
	}, {
		count: 8,
		delta: 1 * Second,
	}} {
		count, delta := test.count, test.delta
		ticker := NewTicker(delta)
		t0 := Now()
		for i := 0; i < count/2; i++ {
			<-ticker.C
		}
		ticker.Reset(delta * 2)
		for i := count / 2; i < count; i++ {
			<-ticker.C
		}
		ticker.Stop()
		t1 := Now()
		dt := t1.Sub(t0)
		target := 3 * delta * Duration(count/2)
		slop := target * 3 / 10
		if dt < target-slop || dt > target+slop {
			errs = append(errs, fmt.Sprintf("%d %s ticks then %d %s ticks took %s, expected [%s,%s]", count/2, delta, count/2, delta*2, dt, target-slop, target+slop))
			if dt > target+slop {
				// System may be overloaded; sleep a bit
				// in the hopes it will recover.
				Sleep(Second / 2)
			}
			continue
		}
		// Now test that the ticker stopped.
		Sleep(2 * delta)
		select {
		case <-ticker.C:
			errs = append(errs, "Ticker did not shut down")
			continue
		default:
			// ok
		}

		// Test passed, so all done.
		if len(errs) > 0 {
			t.Logf("saw %d errors, ignoring to avoid flakiness", len(errs))
			logErrs()
		}

		return
	}

	t.Errorf("saw %d errors", len(errs))
	logErrs()
}

// Issue 21874
func TestTickerStopWithDirectInitialization(t *testing.T) {
	c := make(chan Time)
	tk := &Ticker{C: c}
	tk.Stop()
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

// Test that Ticker.Reset panics when given a duration less than zero.
func TestTickerResetLtZeroDuration(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Errorf("Ticker.Reset(0) should have panicked")
		}
	}()
	tk := NewTicker(Second)
	tk.Reset(0)
}

func TestLongAdjustTimers(t *testing.T) {
	if runtime.GOOS == "android" || runtime.GOOS == "ios" {
		t.Skipf("skipping on %s - too slow", runtime.GOOS)
	}
	t.Parallel()
	var wg sync.WaitGroup
	defer wg.Wait()

	// Build up the timer heap.
	const count = 5000
	wg.Add(count)
	for range count {
		go func() {
			defer wg.Done()
			Sleep(10 * Microsecond)
		}()
	}
	for range count {
		Sleep(1 * Microsecond)
	}

	// Give ourselves 60 seconds to complete.
	// This used to reliably fail on a Mac M3 laptop,
	// which needed 77 seconds.
	// Trybots are slower, so it will fail even more reliably there.
	// With the fix, the code runs in under a second.
	done := make(chan bool)
	AfterFunc(60*Second, func() { close(done) })

	// Set up a queing goroutine to ping pong through the scheduler.
	inQ := make(chan func())
	outQ := make(chan func())

	defer close(inQ)

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(outQ)
		var q []func()
		for {
			var sendTo chan func()
			var send func()
			if len(q) > 0 {
				sendTo = outQ
				send = q[0]
			}
			select {
			case sendTo <- send:
				q = q[1:]
			case f, ok := <-inQ:
				if !ok {
					return
				}
				q = append(q, f)
			case <-done:
				return
			}
		}
	}()

	for i := range 50000 {
		const try = 20
		for range try {
			inQ <- func() {}
		}
		for range try {
			select {
			case _, ok := <-outQ:
				if !ok {
					t.Fatal("output channel is closed")
				}
			case <-After(5 * Second):
				t.Fatalf("failed to read work, iteration %d", i)
			case <-done:
				t.Fatal("timer expired")
			}
		}
	}
}
func BenchmarkTicker(b *testing.B) {
	benchmark(b, func(n int) {
		ticker := NewTicker(Nanosecond)
		for i := 0; i < n; i++ {
			<-ticker.C
		}
		ticker.Stop()
	})
}

func BenchmarkTickerReset(b *testing.B) {
	benchmark(b, func(n int) {
		ticker := NewTicker(Nanosecond)
		for i := 0; i < n; i++ {
			ticker.Reset(Nanosecond * 2)
		}
		ticker.Stop()
	})
}

func BenchmarkTickerResetNaive(b *testing.B) {
	benchmark(b, func(n int) {
		ticker := NewTicker(Nanosecond)
		for i := 0; i < n; i++ {
			ticker.Stop()
			ticker = NewTicker(Nanosecond * 2)
		}
		ticker.Stop()
	})
}
