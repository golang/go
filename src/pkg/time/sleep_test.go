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
	const delay = int64(100e6)
	go func() {
		Sleep(delay / 2)
		Interrupt()
	}()
	start := Nanoseconds()
	Sleep(delay)
	duration := Nanoseconds() - start
	if duration < delay {
		t.Fatalf("Sleep(%d) slept for only %d ns", delay, duration)
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
			Sleep(1e9)
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
		NewTimer(1e9).Stop()
	}
}

func TestAfter(t *testing.T) {
	const delay = int64(100e6)
	start := Nanoseconds()
	end := <-After(delay)
	if duration := Nanoseconds() - start; duration < delay {
		t.Fatalf("After(%d) slept for only %d ns", delay, duration)
	}
	if min := start + delay; end < min {
		t.Fatalf("After(%d) expect >= %d, got %d", delay, min, end)
	}
}

func TestAfterTick(t *testing.T) {
	const (
		Delta = 100 * 1e6
		Count = 10
	)
	t0 := Nanoseconds()
	for i := 0; i < Count; i++ {
		<-After(Delta)
	}
	t1 := Nanoseconds()
	ns := t1 - t0
	target := int64(Delta * Count)
	slop := target * 2 / 10
	if ns < target-slop || ns > target+slop {
		t.Fatalf("%d ticks of %g ns took %g ns, expected %g", Count, float64(Delta), float64(ns), float64(target))
	}
}

func TestAfterStop(t *testing.T) {
	const msec = 1e6
	AfterFunc(100*msec, func() {})
	t0 := NewTimer(50 * msec)
	c1 := make(chan bool, 1)
	t1 := AfterFunc(150*msec, func() { c1 <- true })
	c2 := After(200 * msec)
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
	t    int64
}

func await(slot int, result chan<- afterResult, ac <-chan int64) {
	result <- afterResult{slot, <-ac}
}

func testAfterQueuing(t *testing.T) error {
	const (
		Delta = 100 * 1e6
	)
	// make the result channel buffered because we don't want
	// to depend on channel queueing semantics that might
	// possibly change in the future.
	result := make(chan afterResult, len(slots))

	t0 := Nanoseconds()
	for _, slot := range slots {
		go await(slot, result, After(int64(slot)*Delta))
	}
	sort.Ints(slots)
	for _, slot := range slots {
		r := <-result
		if r.slot != slot {
			return fmt.Errorf("after queue got slot %d, expected %d", r.slot, slot)
		}
		ns := r.t - t0
		target := int64(slot * Delta)
		slop := int64(Delta) / 4
		if ns < target-slop || ns > target+slop {
			return fmt.Errorf("after queue slot %d arrived at %g, expected [%g,%g]", slot, float64(ns), float64(target-slop), float64(target+slop))
		}
	}
	return nil
}
