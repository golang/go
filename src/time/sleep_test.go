// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"errors"
	"fmt"
	"internal/testenv"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	. "time"
	_ "unsafe" // for go:linkname
)

// newTimerFunc simulates NewTimer using AfterFunc,
// but this version will not hit the special cases for channels
// that are used when calling NewTimer.
// This makes it easy to test both paths.
func newTimerFunc(d Duration) *Timer {
	c := make(chan Time, 1)
	t := AfterFunc(d, func() { c <- Now() })
	t.C = c
	return t
}

// haveHighResSleep is true if the system supports at least ~1ms sleeps.
//
//go:linkname haveHighResSleep runtime.haveHighResSleep
var haveHighResSleep bool

// adjustDelay returns an adjusted delay based on the system sleep resolution.
// Go runtime uses different Windows timers for time.Now and sleeping.
// These can tick at different frequencies and can arrive out of sync.
// The effect can be seen, for example, as time.Sleep(100ms) is actually
// shorter then 100ms when measured as difference between time.Now before and
// after time.Sleep call. This was observed on Windows XP SP3 (windows/386).
func adjustDelay(t *testing.T, delay Duration) Duration {
	if haveHighResSleep {
		return delay
	}
	t.Log("adjusting delay for low resolution sleep")
	switch runtime.GOOS {
	case "windows":
		return delay - 17*Millisecond
	default:
		t.Fatal("adjustDelay unimplemented on " + runtime.GOOS)
		return 0
	}
}

func TestSleep(t *testing.T) {
	const delay = 100 * Millisecond
	go func() {
		Sleep(delay / 2)
		Interrupt()
	}()
	start := Now()
	Sleep(delay)
	delayadj := adjustDelay(t, delay)
	duration := Since(start)
	if duration < delayadj {
		t.Fatalf("Sleep(%s) slept for only %s", delay, duration)
	}
}

// Test the basic function calling behavior. Correct queuing
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

func TestTickerStress(t *testing.T) {
	var stop atomic.Bool
	go func() {
		for !stop.Load() {
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
	stop.Store(true)
}

func TestTickerConcurrentStress(t *testing.T) {
	var stop atomic.Bool
	go func() {
		for !stop.Load() {
			runtime.GC()
			// Yield so that the OS can wake up the timer thread,
			// so that it can generate channel sends for the main goroutine,
			// which will eventually set stop = 1 for us.
			Sleep(Nanosecond)
		}
	}()
	ticker := NewTicker(1)
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				<-ticker.C
			}
		}()
	}
	wg.Wait()
	ticker.Stop()
	stop.Store(true)
}

func TestAfterFuncStarvation(t *testing.T) {
	// Start two goroutines ping-ponging on a channel send.
	// At any given time, at least one of these goroutines is runnable:
	// if the channel buffer is full, the receiver is runnable,
	// and if it is not full, the sender is runnable.
	//
	// In addition, the AfterFunc callback should become runnable after
	// the indicated delay.
	//
	// Even if GOMAXPROCS=1, we expect the runtime to eventually schedule
	// the AfterFunc goroutine instead of the runnable channel goroutine.
	// However, in https://go.dev/issue/65178 this was observed to live-lock
	// on wasip1/wasm and js/wasm after <10000 runs.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	var (
		wg   sync.WaitGroup
		stop atomic.Bool
		c    = make(chan bool, 1)
	)

	wg.Add(2)
	go func() {
		for !stop.Load() {
			c <- true
		}
		close(c)
		wg.Done()
	}()
	go func() {
		for range c {
		}
		wg.Done()
	}()

	AfterFunc(1*Microsecond, func() { stop.Store(true) })
	wg.Wait()
}

func benchmark(b *testing.B, bench func(*testing.PB)) {
	// Create equal number of garbage timers on each P before starting
	// the benchmark.
	var wg sync.WaitGroup
	garbageAll := make([][]*Timer, runtime.GOMAXPROCS(0))
	for i := range garbageAll {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			garbage := make([]*Timer, 1<<15)
			for j := range garbage {
				garbage[j] = AfterFunc(Hour, nil)
			}
			garbageAll[i] = garbage
		}(i)
	}
	wg.Wait()

	b.ResetTimer()
	b.RunParallel(bench)
	b.StopTimer()

	for _, garbage := range garbageAll {
		for _, t := range garbage {
			t.Stop()
		}
	}
}

func BenchmarkAfterFunc1000(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		for pb.Next() {
			n := 1000
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
		}
	})
}

func BenchmarkAfter(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		for pb.Next() {
			<-After(1)
		}
	})
}

func BenchmarkStop(b *testing.B) {
	b.Run("impl=chan", func(b *testing.B) {
		benchmark(b, func(pb *testing.PB) {
			for pb.Next() {
				NewTimer(1 * Second).Stop()
			}
		})
	})
	b.Run("impl=func", func(b *testing.B) {
		benchmark(b, func(pb *testing.PB) {
			for pb.Next() {
				newTimerFunc(1 * Second).Stop()
			}
		})
	})
}

func BenchmarkSimultaneousAfterFunc1000(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		for pb.Next() {
			n := 1000
			var wg sync.WaitGroup
			wg.Add(n)
			for range n {
				AfterFunc(0, wg.Done)
			}
			wg.Wait()
		}
	})
}

func BenchmarkStartStop1000(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		for pb.Next() {
			const N = 1000
			timers := make([]*Timer, N)
			for i := range timers {
				timers[i] = AfterFunc(Hour, nil)
			}

			for i := range timers {
				timers[i].Stop()
			}
		}
	})
}

func BenchmarkReset(b *testing.B) {
	b.Run("impl=chan", func(b *testing.B) {
		benchmark(b, func(pb *testing.PB) {
			t := NewTimer(Hour)
			for pb.Next() {
				t.Reset(Hour)
			}
			t.Stop()
		})
	})
	b.Run("impl=func", func(b *testing.B) {
		benchmark(b, func(pb *testing.PB) {
			t := newTimerFunc(Hour)
			for pb.Next() {
				t.Reset(Hour)
			}
			t.Stop()
		})
	})
}

func BenchmarkSleep1000(b *testing.B) {
	benchmark(b, func(pb *testing.PB) {
		for pb.Next() {
			const N = 1000
			var wg sync.WaitGroup
			wg.Add(N)
			for range N {
				go func() {
					Sleep(Nanosecond)
					wg.Done()
				}()
			}
			wg.Wait()
		}
	})
}

func TestAfter(t *testing.T) {
	const delay = 100 * Millisecond
	start := Now()
	end := <-After(delay)
	delayadj := adjustDelay(t, delay)
	if duration := Since(start); duration < delayadj {
		t.Fatalf("After(%s) slept for only %d ns", delay, duration)
	}
	if min := start.Add(delayadj); end.Before(min) {
		t.Fatalf("After(%s) expect >= %s, got %s", delay, min, end)
	}
}

func TestAfterTick(t *testing.T) {
	t.Parallel()
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
	t.Run("impl=chan", func(t *testing.T) {
		testAfterStop(t, NewTimer)
	})
	t.Run("impl=func", func(t *testing.T) {
		testAfterStop(t, newTimerFunc)
	})
}

func testAfterStop(t *testing.T, newTimer func(Duration) *Timer) {
	// We want to test that we stop a timer before it runs.
	// We also want to test that it didn't run after a longer timer.
	// Since we don't want the test to run for too long, we don't
	// want to use lengthy times. That makes the test inherently flaky.
	// So only report an error if it fails five times in a row.

	var errs []string
	logErrs := func() {
		for _, e := range errs {
			t.Log(e)
		}
	}

	for i := 0; i < 5; i++ {
		AfterFunc(100*Millisecond, func() {})
		t0 := newTimer(50 * Millisecond)
		c1 := make(chan bool, 1)
		t1 := AfterFunc(150*Millisecond, func() { c1 <- true })
		c2 := After(200 * Millisecond)
		if !t0.Stop() {
			errs = append(errs, "failed to stop event 0")
			continue
		}
		if !t1.Stop() {
			errs = append(errs, "failed to stop event 1")
			continue
		}
		<-c2
		select {
		case <-t0.C:
			errs = append(errs, "event 0 was not stopped")
			continue
		case <-c1:
			errs = append(errs, "event 1 was not stopped")
			continue
		default:
		}
		if t1.Stop() {
			errs = append(errs, "Stop returned true twice")
			continue
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

func TestAfterQueuing(t *testing.T) {
	t.Run("impl=chan", func(t *testing.T) {
		testAfterQueuing(t, After)
	})
	t.Run("impl=func", func(t *testing.T) {
		testAfterQueuing(t, func(d Duration) <-chan Time { return newTimerFunc(d).C })
	})
}

func testAfterQueuing(t *testing.T, after func(Duration) <-chan Time) {
	// This test flakes out on some systems,
	// so we'll try it a few times before declaring it a failure.
	const attempts = 5
	err := errors.New("!=nil")
	for i := 0; i < attempts && err != nil; i++ {
		delta := Duration(20+i*50) * Millisecond
		if err = testAfterQueuing1(delta, after); err != nil {
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

func testAfterQueuing1(delta Duration, after func(Duration) <-chan Time) error {
	// make the result channel buffered because we don't want
	// to depend on channel queuing semantics that might
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
	t.Parallel()
	for i := 0; i < 100; i++ {
		go func(i int) {
			timer := AfterFunc(2*Second, func() {
				t.Errorf("timer %d was not stopped", i)
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
	if !t0.Reset(3 * d) {
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

	if t0.Reset(50 * Millisecond) {
		return errors.New("resetting expired timer returned true")
	}
	return nil
}

func TestReset(t *testing.T) {
	// We try to run this test with increasingly larger multiples
	// until one works so slow, loaded hardware isn't as flaky,
	// but without slowing down fast machines unnecessarily.
	//
	// (maxDuration is several orders of magnitude longer than we
	// expect this test to actually take on a fast, unloaded machine.)
	d := 1 * Millisecond
	const maxDuration = 10 * Second
	for {
		err := testReset(d)
		if err == nil {
			break
		}
		d *= 2
		if d > maxDuration {
			t.Error(err)
		}
		t.Logf("%v; trying duration %v", err, d)
	}
}

// Test that sleeping (via Sleep or Timer) for an interval so large it
// overflows does not result in a short sleep duration. Nor does it interfere
// with execution of other timers. If it does, timers in this or subsequent
// tests may not fire.
func TestOverflowSleep(t *testing.T) {
	const big = Duration(int64(1<<63 - 1))

	go func() {
		Sleep(big)
		// On failure, this may return after the test has completed, so
		// we need to panic instead.
		panic("big sleep returned")
	}()

	select {
	case <-After(big):
		t.Fatalf("big timeout fired")
	case <-After(25 * Millisecond):
		// OK
	}

	const neg = Duration(-1 << 63)
	Sleep(neg) // Returns immediately.
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

func TestOverflowPeriodRuntimeTimer(t *testing.T) {
	// This may hang forever if timers are broken. See comment near
	// the end of CheckRuntimeTimerOverflow in internal_test.go.
	CheckRuntimeTimerPeriodOverflow()
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

// Test that zero duration timers aren't missed by the scheduler. Regression test for issue 44868.
func TestZeroTimer(t *testing.T) {
	t.Run("impl=chan", func(t *testing.T) {
		testZeroTimer(t, NewTimer)
	})
	t.Run("impl=func", func(t *testing.T) {
		testZeroTimer(t, newTimerFunc)
	})
	t.Run("impl=cache", func(t *testing.T) {
		timer := newTimerFunc(Hour)
		testZeroTimer(t, func(d Duration) *Timer {
			timer.Reset(d)
			return timer
		})
	})
}

func testZeroTimer(t *testing.T, newTimer func(Duration) *Timer) {
	if testing.Short() {
		t.Skip("-short")
	}

	for i := 0; i < 1000000; i++ {
		s := Now()
		ti := newTimer(0)
		<-ti.C
		if diff := Since(s); diff > 2*Second {
			t.Errorf("Expected time to get value from Timer channel in less than 2 sec, took %v", diff)
		}
	}
}

// Test that rapidly moving a timer earlier doesn't cause it to get dropped.
// Issue 47329.
func TestTimerModifiedEarlier(t *testing.T) {
	if runtime.GOOS == "plan9" && runtime.GOARCH == "arm" {
		testenv.SkipFlaky(t, 50470)
	}

	past := Until(Unix(0, 0))
	count := 1000
	fail := 0
	for i := 0; i < count; i++ {
		timer := newTimerFunc(Hour)
		for j := 0; j < 10; j++ {
			if !timer.Stop() {
				<-timer.C
			}
			timer.Reset(past)
		}

		deadline := NewTimer(10 * Second)
		defer deadline.Stop()
		now := Now()
		select {
		case <-timer.C:
			if since := Since(now); since > 8*Second {
				t.Errorf("timer took too long (%v)", since)
				fail++
			}
		case <-deadline.C:
			t.Error("deadline expired")
		}
	}

	if fail > 0 {
		t.Errorf("%d failures", fail)
	}
}

// Test that rapidly moving timers earlier and later doesn't cause
// some of the sleep times to be lost.
// Issue 47762
func TestAdjustTimers(t *testing.T) {
	var rnd = rand.New(rand.NewSource(Now().UnixNano()))

	timers := make([]*Timer, 100)
	states := make([]int, len(timers))
	indices := rnd.Perm(len(timers))

	for len(indices) != 0 {
		var ii = rnd.Intn(len(indices))
		var i = indices[ii]

		var timer = timers[i]
		var state = states[i]
		states[i]++

		switch state {
		case 0:
			timers[i] = newTimerFunc(0)

		case 1:
			<-timer.C // Timer is now idle.

		// Reset to various long durations, which we'll cancel.
		case 2:
			if timer.Reset(1 * Minute) {
				panic("shouldn't be active (1)")
			}
		case 4:
			if timer.Reset(3 * Minute) {
				panic("shouldn't be active (3)")
			}
		case 6:
			if timer.Reset(2 * Minute) {
				panic("shouldn't be active (2)")
			}

		// Stop and drain a long-duration timer.
		case 3, 5, 7:
			if !timer.Stop() {
				t.Logf("timer %d state %d Stop returned false", i, state)
				<-timer.C
			}

		// Start a short-duration timer we expect to select without blocking.
		case 8:
			if timer.Reset(0) {
				t.Fatal("timer.Reset returned true")
			}
		case 9:
			now := Now()
			<-timer.C
			dur := Since(now)
			if dur > 750*Millisecond {
				t.Errorf("timer %d took %v to complete", i, dur)
			}

		// Timer is done. Swap with tail and remove.
		case 10:
			indices[ii] = indices[len(indices)-1]
			indices = indices[:len(indices)-1]
		}
	}
}

// Benchmark timer latency when the thread that creates the timer is busy with
// other work and the timers must be serviced by other threads.
// https://golang.org/issue/38860
func BenchmarkParallelTimerLatency(b *testing.B) {
	gmp := runtime.GOMAXPROCS(0)
	if gmp < 2 || runtime.NumCPU() < gmp {
		b.Skip("skipping with GOMAXPROCS < 2 or NumCPU < GOMAXPROCS")
	}

	// allocate memory now to avoid GC interference later.
	timerCount := gmp - 1
	stats := make([]struct {
		sum   float64
		max   Duration
		count int64
		_     [5]int64 // cache line padding
	}, timerCount)

	// Ensure the time to start new threads to service timers will not pollute
	// the results.
	warmupScheduler(gmp)

	// Note that other than the AfterFunc calls this benchmark is measuring it
	// avoids using any other timers. In particular, the main goroutine uses
	// doWork to spin for some durations because up through Go 1.15 if all
	// threads are idle sysmon could leave deep sleep when we wake.

	// Ensure sysmon is in deep sleep.
	doWork(30 * Millisecond)

	b.ResetTimer()

	const delay = Millisecond
	var wg sync.WaitGroup
	var count int32
	for i := 0; i < b.N; i++ {
		wg.Add(timerCount)
		atomic.StoreInt32(&count, 0)
		for j := 0; j < timerCount; j++ {
			j := j
			expectedWakeup := Now().Add(delay)
			AfterFunc(delay, func() {
				late := Since(expectedWakeup)
				if late < 0 {
					late = 0
				}
				stats[j].count++
				stats[j].sum += float64(late.Nanoseconds())
				if late > stats[j].max {
					stats[j].max = late
				}
				atomic.AddInt32(&count, 1)
				for atomic.LoadInt32(&count) < int32(timerCount) {
					// spin until all timers fired
				}
				wg.Done()
			})
		}

		for atomic.LoadInt32(&count) < int32(timerCount) {
			// spin until all timers fired
		}
		wg.Wait()

		// Spin for a bit to let the other scheduler threads go idle before the
		// next round.
		doWork(Millisecond)
	}
	var total float64
	var samples float64
	max := Duration(0)
	for _, s := range stats {
		if s.max > max {
			max = s.max
		}
		total += s.sum
		samples += float64(s.count)
	}
	b.ReportMetric(0, "ns/op")
	b.ReportMetric(total/samples, "avg-late-ns")
	b.ReportMetric(float64(max.Nanoseconds()), "max-late-ns")
}

// Benchmark timer latency with staggered wakeup times and varying CPU bound
// workloads. https://golang.org/issue/38860
func BenchmarkStaggeredTickerLatency(b *testing.B) {
	gmp := runtime.GOMAXPROCS(0)
	if gmp < 2 || runtime.NumCPU() < gmp {
		b.Skip("skipping with GOMAXPROCS < 2 or NumCPU < GOMAXPROCS")
	}

	const delay = 3 * Millisecond

	for _, dur := range []Duration{300 * Microsecond, 2 * Millisecond} {
		b.Run(fmt.Sprintf("work-dur=%s", dur), func(b *testing.B) {
			for tickersPerP := 1; tickersPerP < int(delay/dur)+1; tickersPerP++ {
				tickerCount := gmp * tickersPerP
				b.Run(fmt.Sprintf("tickers-per-P=%d", tickersPerP), func(b *testing.B) {
					// allocate memory now to avoid GC interference later.
					stats := make([]struct {
						sum   float64
						max   Duration
						count int64
						_     [5]int64 // cache line padding
					}, tickerCount)

					// Ensure the time to start new threads to service timers
					// will not pollute the results.
					warmupScheduler(gmp)

					b.ResetTimer()

					var wg sync.WaitGroup
					wg.Add(tickerCount)
					for j := 0; j < tickerCount; j++ {
						j := j
						doWork(delay / Duration(gmp))
						expectedWakeup := Now().Add(delay)
						ticker := NewTicker(delay)
						go func(c int, ticker *Ticker, firstWake Time) {
							defer ticker.Stop()

							for ; c > 0; c-- {
								<-ticker.C
								late := Since(expectedWakeup)
								if late < 0 {
									late = 0
								}
								stats[j].count++
								stats[j].sum += float64(late.Nanoseconds())
								if late > stats[j].max {
									stats[j].max = late
								}
								expectedWakeup = expectedWakeup.Add(delay)
								doWork(dur)
							}
							wg.Done()
						}(b.N, ticker, expectedWakeup)
					}
					wg.Wait()

					var total float64
					var samples float64
					max := Duration(0)
					for _, s := range stats {
						if s.max > max {
							max = s.max
						}
						total += s.sum
						samples += float64(s.count)
					}
					b.ReportMetric(0, "ns/op")
					b.ReportMetric(total/samples, "avg-late-ns")
					b.ReportMetric(float64(max.Nanoseconds()), "max-late-ns")
				})
			}
		})
	}
}

// warmupScheduler ensures the scheduler has at least targetThreadCount threads
// in its thread pool.
func warmupScheduler(targetThreadCount int) {
	var wg sync.WaitGroup
	var count int32
	for i := 0; i < targetThreadCount; i++ {
		wg.Add(1)
		go func() {
			atomic.AddInt32(&count, 1)
			for atomic.LoadInt32(&count) < int32(targetThreadCount) {
				// spin until all threads started
			}

			// spin a bit more to ensure they are all running on separate CPUs.
			doWork(Millisecond)
			wg.Done()
		}()
	}
	wg.Wait()
}

func doWork(dur Duration) {
	start := Now()
	for Since(start) < dur {
	}
}
