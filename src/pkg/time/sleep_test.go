// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync/atomic"
	"testing"
	. "time"
)

func TestSleep(t *testing.T) {
	const delay = 100 * Millisecond
	go func() {
		Sleep(delay / 2)
		Interrupt()
	}()
	start := Now()
	Sleep(delay)
	duration := Now().Sub(start)
	if duration < delay {
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
			// Need to yield, because otherwise
			// the main goroutine will never set the stop flag.
			runtime.Gosched()
		}
	}()
	c := Tick(1)
	for i := 0; i < 100; i++ {
		<-c
	}
	atomic.StoreUint32(&stop, 1)
}

func BenchmarkAfterFunc(b *testing.B) {
	i := b.N
	c := make(chan bool)
	var f func()
	f = func() {
		i--
		if i >= 0 {
			AfterFunc(0, f)
		} else {
			c <- true
		}
	}

	AfterFunc(0, f)
	<-c
}

func BenchmarkAfter(b *testing.B) {
	for i := 0; i < b.N; i++ {
		<-After(1)
	}
}

func BenchmarkStop(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NewTimer(1 * Second).Stop()
	}
}

func TestAfter(t *testing.T) {
	const delay = 100 * Millisecond
	start := Now()
	end := <-After(delay)
	if duration := Now().Sub(start); duration < delay {
		t.Fatalf("After(%s) slept for only %d ns", delay, duration)
	}
	if min := start.Add(delay); end.Before(min) {
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
	if d < target*9/10 || d > target*30/10 {
		t.Fatalf("%d ticks of %s took %s, expected %s", Count, Delta, d, target)
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
	const attempts = 3
	err := errors.New("!=nil")
	for i := 0; i < attempts && err != nil; i++ {
		if err = testAfterQueuing(t); err != nil {
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

func testAfterQueuing(t *testing.T) error {
	Delta := 100 * Millisecond
	if testing.Short() {
		Delta = 20 * Millisecond
	}
	// make the result channel buffered because we don't want
	// to depend on channel queueing semantics that might
	// possibly change in the future.
	result := make(chan afterResult, len(slots))

	t0 := Now()
	for _, slot := range slots {
		go await(slot, result, After(Duration(slot)*Delta))
	}
	sort.Ints(slots)
	for _, slot := range slots {
		r := <-result
		if r.slot != slot {
			return fmt.Errorf("after slot %d, expected %d", r.slot, slot)
		}
		dt := r.t.Sub(t0)
		target := Duration(slot) * Delta
		if dt < target-Delta/2 || dt > target+Delta*10 {
			return fmt.Errorf("After(%s) arrived at %s, expected [%s,%s]", target, dt, target-Delta/2, target+Delta*10)
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
