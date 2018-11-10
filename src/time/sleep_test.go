// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	. "time"
)

// Go runtime uses different Windows timers for time.Now and sleeping.
// These can tick at different frequencies and can arrive out of sync.
// The effect can be seen, for example, as time.Sleep(100ms) is actually
// shorter then 100ms when measured as difference between time.Now before and
// after time.Sleep call. This was observed on Windows XP SP3 (windows/386).
// windowsInaccuracy is to ignore such errors.
const windowsInaccuracy = 17 * Millisecond

func TestSleep(t *testing.T) {
	const delay = 100 * Millisecond
	go func() {
		Sleep(delay / 2)
		Interrupt()
	}()
	start := Now()
	Sleep(delay)
	delayadj := delay
	if runtime.GOOS == "windows" {
		delayadj -= windowsInaccuracy
	}
	duration := Now().Sub(start)
	if duration < delayadj {
		t.Fatalf("Sleep(%s) slept for only %s", delay, duration)
	}
}

// Test the basic function calling behavior. Correct queueing
// behavior is tested elsewhere, since After and AfterFunc share
// the same code.
func TestAfterFunc(t *testing.T) {
	i := 10
	c := make(chan bool)
	var f func()
	f = func() {
		i--
		if i >= 0 {
			AfterFunc(0, f)
			Sleep(1 * Second)
		} else {
			c <- true
		}
	}

	AfterFunc(0, f)
	<-c
}

func TestAfterStress(t *testing.T) {
	stop := uint32(0)
	go func() {
		for atomic.LoadUint32(&stop) == 0 {
			runtime.GC()
			// Yield so that the OS can wake up the timer thread,
			// so that it can generate channel sends for the main goroutine,
			// which will eventually set stop = 1 for us.
			Sleep(Nanosecond)
		}
	}()
	ticker := NewTicker(1)
	for i := 0; i < 100; i++ {
		<-ticker.C
	}
	ticker.Stop()
	atomic.StoreUint32(&stop, 1)
}

func benchmark(b *testing.B, bench func(n int)) {
	garbage := make([]*Timer, 1<<17)
	for i := 0; i < len(garbage); i++ {
		garbage[i] = AfterFunc(Hour, nil)
	}
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			bench(1000)
		}
	})

	b.StopTimer()
	for i := 0; i < len(garbage); i++ {
		garbage[i].Stop()
	}
}

func BenchmarkAfterFunc(b *testing.B) {
	benchmark(b, func(n int) {
		c := make(chan bool)
		var f func()
		f = func() {
			n--
			if n >= 0 {
				AfterFunc(0, f)
			} else {
				c <- true
			}
		}

		AfterFunc(0, f)
		<-c
	})
}

func BenchmarkAfter(b *testing.B) {
	benchmark(b, func(n int) {
		for i := 0; i < n; i++ {
			<-After(1)
		}
	})
}

func BenchmarkStop(b *testing.B) {
	benchmark(b, func(n int) {
		for i := 0; i < n; i++ {
			NewTimer(1 * Second).Stop()
		}
	})
}

func BenchmarkSimultaneousAfterFunc(b *testing.B) {
	benchmark(b, func(n int) {
		var wg sync.WaitGroup
		wg.Add(n)
		for i := 0; i < n; i++ {
			AfterFunc(0, wg.Done)
		}
		wg.Wait()
	})
}

func BenchmarkStartStop(b *testing.B) {
	benchmark(b, func(n int) {
		timers := make([]*Timer, n)
		for i := 0; i < n; i++ {
			timers[i] = AfterFunc(Hour, nil)
		}

		for i := 0; i < n; i++ {
			timers[i].Stop()
		}
	})
}

func TestAfter(t *testing.T) {
	const delay = 100 * Millisecond
	start := Now()
	end := <-After(delay)
	delayadj := delay
	if runtime.GOOS == "windows" {
		delayadj -= windowsInaccuracy
	}
	if duration := Now().Sub(start); duration < delayadj {
		t.Fatalf("After(%s) slept for only %d ns", delay, duration)
	}
	if min := start.Add(delayadj); end.Before(min) {
		t.Fatalf("After(%s) expect >= %s, got %s", delay, min, end)
	}
}

func TestAfterTick(t *testing.T) {
	const Count = 10
	Delta := 100 * Millisecond
	if testing.Short() {
		Delta = 10 * Millisecond
	}
	t0 := Now()
	for i := 0; i < Count; i++ {
		<-After(Delta)
	}
	t1 := Now()
	d := t1.Sub(t0)
	target := Delta * Count
	if d < target*9/10 {
		t.Fatalf("%d ticks of %s too fast: took %s, expected %s", Count, Delta, d, target)
	}
	if !testing.Short() && d > target*30/10 {
		t.Fatalf("%d ticks of %s too slow: took %s, expected %s", Count, Delta, d, target)
	}
}

func TestAfterStop(t *testing.T) {
	AfterFunc(100*Millisecond, func() {})
	t0 := NewTimer(50 * Millisecond)
	c1 := make(chan bool, 1)
	t1 := AfterFunc(150*Millisecond, func() { c1 <- true })
	c2 := After(200 * Millisecond)
	if !t0.Stop() {
		t.Fatalf("failed to stop event 0")
	}
	if !t1.Stop() {
		t.Fatalf("failed to stop event 1")
	}
	<-c2
	select {
	case <-t0.C:
		t.Fatalf("event 0 was not stopped")
	case <-c1:
		t.Fatalf("event 1 was not stopped")
	default:
	}
	if t1.Stop() {
		t.Fatalf("Stop returned true twice")
	}
}

func TestAfterQueuing(t *testing.T) {
	// This test flakes out on some systems,
	// so we'll try it a few times before declaring it a failure.
	const attempts = 5
	err := errors.New("!=nil")
	for i := 0; i < attempts && err != nil; i++ {
		delta := Duration(20+i*50) * Millisecond
		if err = testAfterQueuing(delta); err != nil {
			t.Logf("attempt %v failed: %v", i, err)
		}
	}
	if err != nil {
		t.Fatal(err)
	}
}

var slots = []int{5, 3, 6, 6, 6, 1, 1, 2, 7, 9, 4, 8, 0}

type afterResult struct {
	slot int
	t    Time
}

func await(slot int, result chan<- afterResult, ac <-chan Time) {
	result <- afterResult{slot, <-ac}
}

func testAfterQueuing(delta Duration) error {
	// make the result channel buffered because we don't want
	// to depend on channel queueing semantics that might
	// possibly change in the future.
	result := make(chan afterResult, len(slots))

	t0 := Now()
	for _, slot := range slots {
		go await(slot, result, After(Duration(slot)*delta))
	}
	var order []int
	var times []Time
	for range slots {
		r := <-result
		order = append(order, r.slot)
		times = append(times, r.t)
	}
	for i := range order {
		if i > 0 && order[i] < order[i-1] {
			return fmt.Errorf("After calls returned out of order: %v", order)
		}
	}
	for i, t := range times {
		dt := t.Sub(t0)
		target := Duration(order[i]) * delta
		if dt < target-delta/2 || dt > target+delta*10 {
			return fmt.Errorf("After(%s) arrived at %s, expected [%s,%s]", target, dt, target-delta/2, target+delta*10)
		}
	}
	return nil
}

func TestTimerStopStress(t *testing.T) {
	if testing.Short() {
		return
	}
	for i := 0; i < 100; i++ {
		go func(i int) {
			timer := AfterFunc(2*Second, func() {
				t.Fatalf("timer %d was not stopped", i)
			})
			Sleep(1 * Second)
			timer.Stop()
		}(i)
	}
	Sleep(3 * Second)
}

func TestSleepZeroDeadlock(t *testing.T) {
	// Sleep(0) used to hang, the sequence of events was as follows.
	// Sleep(0) sets G's status to Gwaiting, but then immediately returns leaving the status.
	// Then the goroutine calls e.g. new and falls down into the scheduler due to pending GC.
	// After the GC nobody wakes up the goroutine from Gwaiting status.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	c := make(chan bool)
	go func() {
		for i := 0; i < 100; i++ {
			runtime.GC()
		}
		c <- true
	}()
	for i := 0; i < 100; i++ {
		Sleep(0)
		tmp := make(chan bool, 1)
		tmp <- true
		<-tmp
	}
	<-c
}

func testReset(d Duration) error {
	t0 := NewTimer(2 * d)
	Sleep(d)
	if t0.Reset(3*d) != true {
		return errors.New("resetting unfired timer returned false")
	}
	Sleep(2 * d)
	select {
	case <-t0.C:
		return errors.New("timer fired early")
	default:
	}
	Sleep(2 * d)
	select {
	case <-t0.C:
	default:
		return errors.New("reset timer did not fire")
	}

	if t0.Reset(50*Millisecond) != false {
		return errors.New("resetting expired timer returned true")
	}
	return nil
}

func TestReset(t *testing.T) {
	// We try to run this test with increasingly larger multiples
	// until one works so slow, loaded hardware isn't as flaky,
	// but without slowing down fast machines unnecessarily.
	const unit = 25 * Millisecond
	tries := []Duration{
		1 * unit,
		3 * unit,
		7 * unit,
		15 * unit,
	}
	var err error
	for _, d := range tries {
		err = testReset(d)
		if err == nil {
			t.Logf("passed using duration %v", d)
			return
		}
	}
	t.Error(err)
}

// Test that sleeping for an interval so large it overflows does not
// result in a short sleep duration.
func TestOverflowSleep(t *testing.T) {
	const big = Duration(int64(1<<63 - 1))
	select {
	case <-After(big):
		t.Fatalf("big timeout fired")
	case <-After(25 * Millisecond):
		// OK
	}
	const neg = Duration(-1 << 63)
	select {
	case <-After(neg):
		// OK
	case <-After(1 * Second):
		t.Fatalf("negative timeout didn't fire")
	}
}

// Test that a panic while deleting a timer does not leave
// the timers mutex held, deadlocking a ticker.Stop in a defer.
func TestIssue5745(t *testing.T) {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm" {
		t.Skipf("skipping on %s/%s, see issue 10043", runtime.GOOS, runtime.GOARCH)
	}

	ticker := NewTicker(Hour)
	defer func() {
		// would deadlock here before the fix due to
		// lock taken before the segfault.
		ticker.Stop()

		if r := recover(); r == nil {
			t.Error("Expected panic, but none happened.")
		}
	}()

	// cause a panic due to a segfault
	var timer *Timer
	timer.Stop()
	t.Error("Should be unreachable.")
}

func TestOverflowRuntimeTimer(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode, see issue 6874")
	}
	// This may hang forever if timers are broken. See comment near
	// the end of CheckRuntimeTimerOverflow in internal_test.go.
	CheckRuntimeTimerOverflow()
}

func checkZeroPanicString(t *testing.T) {
	e := recover()
	s, _ := e.(string)
	if want := "called on uninitialized Timer"; !strings.Contains(s, want) {
		t.Errorf("panic = %v; want substring %q", e, want)
	}
}

func TestZeroTimerResetPanics(t *testing.T) {
	defer checkZeroPanicString(t)
	var tr Timer
	tr.Reset(1)
}

func TestZeroTimerStopPanics(t *testing.T) {
	defer checkZeroPanicString(t)
	var tr Timer
	tr.Stop()
}
