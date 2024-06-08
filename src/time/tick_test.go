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
	t.Parallel()

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
		for range count / 2 {
			<-ticker.C
		}
		ticker.Reset(delta * 2)
		for range count - count/2 {
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
	t.Parallel()

	Delta := 100 * Millisecond
	if testing.Short() {
		Delta = 20 * Millisecond
	}
	for range 3 {
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

	// Set up a queuing goroutine to ping pong through the scheduler.
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
	benchmark(b, func(pb *testing.PB) {
		ticker := NewTicker(Nanosecond)
		for pb.Next() {
			<-ticker.C
		}
		ticker.Stop()
	})
}

func BenchmarkTickerReset(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		ticker := NewTicker(Nanosecond)
		for pb.Next() {
			ticker.Reset(Nanosecond * 2)
		}
		ticker.Stop()
	})
}

func BenchmarkTickerResetNaive(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		ticker := NewTicker(Nanosecond)
		for pb.Next() {
			ticker.Stop()
			ticker = NewTicker(Nanosecond * 2)
		}
		ticker.Stop()
	})
}

func TestTimerGC(t *testing.T) {
	run := func(t *testing.T, what string, f func()) {
		t.Helper()
		t.Run(what, func(t *testing.T) {
			t.Helper()
			const N = 1e4
			var stats runtime.MemStats
			runtime.GC()
			runtime.GC()
			runtime.GC()
			runtime.ReadMemStats(&stats)
			before := int64(stats.Mallocs - stats.Frees)

			for j := 0; j < N; j++ {
				f()
			}

			runtime.GC()
			runtime.GC()
			runtime.GC()
			runtime.ReadMemStats(&stats)
			after := int64(stats.Mallocs - stats.Frees)

			// Allow some slack, but inuse >= N means at least 1 allocation per iteration.
			inuse := after - before
			if inuse >= N {
				t.Errorf("%s did not get GC'ed: %d allocations", what, inuse)

				Sleep(1 * Second)
				runtime.ReadMemStats(&stats)
				after := int64(stats.Mallocs - stats.Frees)
				inuse = after - before
				t.Errorf("after a sleep: %d allocations", inuse)
			}
		})
	}

	run(t, "After", func() { After(Hour) })
	run(t, "Tick", func() { Tick(Hour) })
	run(t, "NewTimer", func() { NewTimer(Hour) })
	run(t, "NewTicker", func() { NewTicker(Hour) })
	run(t, "NewTimerStop", func() { NewTimer(Hour).Stop() })
	run(t, "NewTickerStop", func() { NewTicker(Hour).Stop() })
}

func TestChan(t *testing.T) {
	for _, name := range []string{"0", "1", "2"} {
		t.Run("asynctimerchan="+name, func(t *testing.T) {
			t.Setenv("GODEBUG", "asynctimerchan="+name)
			t.Run("Timer", func(t *testing.T) {
				tim := NewTimer(10000 * Second)
				testTimerChan(t, tim, tim.C, name == "0")
			})
			t.Run("Ticker", func(t *testing.T) {
				tim := &tickerTimer{Ticker: NewTicker(10000 * Second)}
				testTimerChan(t, tim, tim.C, name == "0")
			})
		})
	}
}

type timer interface {
	Stop() bool
	Reset(Duration) bool
}

// tickerTimer is a Timer with Reset and Stop methods that return bools,
// to have the same signatures as Timer.
type tickerTimer struct {
	*Ticker
	stopped bool
}

func (t *tickerTimer) Stop() bool {
	pending := !t.stopped
	t.stopped = true
	t.Ticker.Stop()
	return pending
}

func (t *tickerTimer) Reset(d Duration) bool {
	pending := !t.stopped
	t.stopped = false
	t.Ticker.Reset(d)
	return pending
}

func testTimerChan(t *testing.T, tim timer, C <-chan Time, synctimerchan bool) {
	_, isTimer := tim.(*Timer)
	isTicker := !isTimer

	// Retry parameters. Enough to deflake even on slow machines.
	// Windows in particular has very coarse timers so we have to
	// wait 10ms just to make a timer go off.
	const (
		sched      = 10 * Millisecond
		tries      = 100
		drainTries = 5
	)

	// drain1 removes one potential stale time value
	// from the timer/ticker channel after Reset.
	// When using Go 1.23 sync timers/tickers, draining is never needed
	// (that's the whole point of the sync timer/ticker change).
	drain1 := func() {
		for range drainTries {
			select {
			case <-C:
				return
			default:
			}
			Sleep(sched)
		}
	}

	// drainAsync removes potential stale time values after Stop/Reset.
	// When using Go 1 async timers, draining one or two values
	// may be needed after Reset or Stop (see comments in body for details).
	drainAsync := func() {
		if synctimerchan {
			// sync timers must have the right semantics without draining:
			// there are no stale values.
			return
		}

		// async timers can send one stale value (then the timer is disabled).
		drain1()
		if isTicker {
			// async tickers can send two stale values: there may be one
			// sitting in the channel buffer, and there may also be one
			// send racing with the Reset/Stop+drain that arrives after
			// the first drain1 has pulled the value out.
			// This is rare, but it does happen on overloaded builder machines.
			// It can also be reproduced on an M3 MacBook Pro using:
			//
			//	go test -c strings
			//	stress ./strings.test &   # chew up CPU
			//	go test -c -race time
			//	stress -p 48 ./time.test -test.count=10 -test.run=TestChan/asynctimerchan=1/Ticker
			drain1()
		}
	}
	noTick := func() {
		t.Helper()
		select {
		default:
		case <-C:
			t.Errorf("extra tick")
		}
	}
	assertTick := func() {
		t.Helper()
		select {
		default:
		case <-C:
			return
		}
		for range tries {
			Sleep(sched)
			select {
			default:
			case <-C:
				return
			}
		}
		t.Errorf("missing tick")
	}
	assertLen := func() {
		t.Helper()
		if synctimerchan {
			if n := len(C); n != 0 {
				t.Errorf("synctimer has len(C) = %d, want 0 (always)", n)
			}
			return
		}
		var n int
		if n = len(C); n == 1 {
			return
		}
		for range tries {
			Sleep(sched)
			if n = len(C); n == 1 {
				return
			}
		}
		t.Errorf("len(C) = %d, want 1", n)
	}

	// Test simple stop; timer never in heap.
	tim.Stop()
	noTick()

	// Test modify of timer not in heap.
	tim.Reset(10000 * Second)
	noTick()

	if synctimerchan {
		// Test modify of timer in heap.
		tim.Reset(1)
		Sleep(sched)
		if l, c := len(C), cap(C); l != 0 || c != 0 {
			//t.Fatalf("len(C), cap(C) = %d, %d, want 0, 0", l, c)
		}
		assertTick()
	} else {
		// Test modify of timer in heap.
		tim.Reset(1)
		assertTick()
		Sleep(sched)
		tim.Reset(10000 * Second)
		drainAsync()
		noTick()

		// Test that len sees an immediate tick arrive
		// for Reset of timer in heap.
		tim.Reset(1)
		assertLen()
		assertTick()

		// Test that len sees an immediate tick arrive
		// for Reset of timer NOT in heap.
		tim.Stop()
		drainAsync()
		tim.Reset(1)
		assertLen()
		assertTick()
	}

	// Sleep long enough that a second tick must happen if this is a ticker.
	// Test that Reset does not lose the tick that should have happened.
	Sleep(sched)
	tim.Reset(10000 * Second)
	drainAsync()
	noTick()

	notDone := func(done chan bool) {
		t.Helper()
		select {
		default:
		case <-done:
			t.Fatalf("early done")
		}
	}

	waitDone := func(done chan bool) {
		t.Helper()
		for range tries {
			Sleep(sched)
			select {
			case <-done:
				return
			default:
			}
		}
		t.Fatalf("never got done")
	}

	// Reset timer in heap (already reset above, but just in case).
	tim.Reset(10000 * Second)
	drainAsync()

	// Test stop while timer in heap (because goroutine is blocked on <-C).
	done := make(chan bool)
	notDone(done)
	go func() {
		<-C
		close(done)
	}()
	Sleep(sched)
	notDone(done)

	// Test reset far away while timer in heap.
	tim.Reset(20000 * Second)
	Sleep(sched)
	notDone(done)

	// Test imminent reset while in heap.
	tim.Reset(1)
	waitDone(done)

	// If this is a ticker, another tick should have come in already
	// (they are 1ns apart). If a timer, it should have stopped.
	if isTicker {
		assertTick()
	} else {
		noTick()
	}

	tim.Stop()
	drainAsync()
	noTick()

	// Again using select and with two goroutines waiting.
	tim.Reset(10000 * Second)
	drainAsync()
	done = make(chan bool, 2)
	done1 := make(chan bool)
	done2 := make(chan bool)
	stop := make(chan bool)
	go func() {
		select {
		case <-C:
			done <- true
		case <-stop:
		}
		close(done1)
	}()
	go func() {
		select {
		case <-C:
			done <- true
		case <-stop:
		}
		close(done2)
	}()
	Sleep(sched)
	notDone(done)
	tim.Reset(sched / 2)
	Sleep(sched)
	waitDone(done)
	tim.Stop()
	close(stop)
	waitDone(done1)
	waitDone(done2)
	if isTicker {
		// extra send might have sent done again
		// (handled by buffering done above).
		select {
		default:
		case <-done:
		}
		// extra send after that might have filled C.
		select {
		default:
		case <-C:
		}
	}
	notDone(done)

	// Test enqueueTimerChan when timer is stopped.
	stop = make(chan bool)
	done = make(chan bool, 2)
	for range 2 {
		go func() {
			select {
			case <-C:
				panic("unexpected data")
			case <-stop:
			}
			done <- true
		}()
	}
	Sleep(sched)
	close(stop)
	waitDone(done)
	waitDone(done)

	// Test that Stop and Reset block old values from being received.
	// (Proposal go.dev/issue/37196.)
	if synctimerchan {
		tim.Reset(1)
		Sleep(10 * Millisecond)
		if pending := tim.Stop(); pending != true {
			t.Errorf("tim.Stop() = %v, want true", pending)
		}
		noTick()

		tim.Reset(Hour)
		noTick()
		if pending := tim.Reset(1); pending != true {
			t.Errorf("tim.Stop() = %v, want true", pending)
		}
		assertTick()
		Sleep(10 * Millisecond)
		if isTicker {
			assertTick()
			Sleep(10 * Millisecond)
		} else {
			noTick()
		}
		if pending, want := tim.Reset(Hour), isTicker; pending != want {
			t.Errorf("tim.Stop() = %v, want %v", pending, want)
		}
		noTick()
	}
}

func TestManualTicker(t *testing.T) {
	// Code should not do this, but some old code dating to Go 1.9 does.
	// Make sure this doesn't crash.
	// See go.dev/issue/21874.
	c := make(chan Time)
	tick := &Ticker{C: c}
	tick.Stop()
}

func TestAfterTimes(t *testing.T) {
	t.Parallel()
	// Using After(10ms) but waiting for 500ms to read the channel
	// should produce a time from start+10ms, not start+500ms.
	// Make sure it does.
	// To avoid flakes due to very long scheduling delays,
	// require 10 failures in a row before deciding something is wrong.
	for range 10 {
		start := Now()
		c := After(10 * Millisecond)
		Sleep(500 * Millisecond)
		dt := (<-c).Sub(start)
		if dt < 400*Millisecond {
			return
		}
		t.Logf("After(10ms) time is +%v, want <400ms", dt)
	}
	t.Errorf("not working")
}

func TestTickTimes(t *testing.T) {
	t.Parallel()
	// See comment in TestAfterTimes
	for range 10 {
		start := Now()
		c := Tick(10 * Millisecond)
		Sleep(500 * Millisecond)
		dt := (<-c).Sub(start)
		if dt < 400*Millisecond {
			return
		}
		t.Logf("Tick(10ms) time is +%v, want <400ms", dt)
	}
	t.Errorf("not working")
}
