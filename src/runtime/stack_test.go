// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestStackMem measures per-thread stack segment cache behavior.
// The test consumed up to 500MB in the past.
func TestStackMem(t *testing.T) {
	const (
		BatchSize      = 32
		BatchCount     = 256
		ArraySize      = 1024
		RecursionDepth = 128
	)
	if testing.Short() {
		return
	}
	defer GOMAXPROCS(GOMAXPROCS(BatchSize))
	s0 := new(MemStats)
	ReadMemStats(s0)
	for b := 0; b < BatchCount; b++ {
		c := make(chan bool, BatchSize)
		for i := 0; i < BatchSize; i++ {
			go func() {
				var f func(k int, a [ArraySize]byte)
				f = func(k int, a [ArraySize]byte) {
					if k == 0 {
						time.Sleep(time.Millisecond)
						return
					}
					f(k-1, a)
				}
				f(RecursionDepth, [ArraySize]byte{})
				c <- true
			}()
		}
		for i := 0; i < BatchSize; i++ {
			<-c
		}

		// The goroutines have signaled via c that they are ready to exit.
		// Give them a chance to exit by sleeping. If we don't wait, we
		// might not reuse them on the next batch.
		time.Sleep(10 * time.Millisecond)
	}
	s1 := new(MemStats)
	ReadMemStats(s1)
	consumed := int64(s1.StackSys - s0.StackSys)
	t.Logf("Consumed %vMB for stack mem", consumed>>20)
	estimate := int64(8 * BatchSize * ArraySize * RecursionDepth) // 8 is to reduce flakiness.
	if consumed > estimate {
		t.Fatalf("Stack mem: want %v, got %v", estimate, consumed)
	}
	// Due to broken stack memory accounting (http://golang.org/issue/7468),
	// StackInuse can decrease during function execution, so we cast the values to int64.
	inuse := int64(s1.StackInuse) - int64(s0.StackInuse)
	t.Logf("Inuse %vMB for stack mem", inuse>>20)
	if inuse > 4<<20 {
		t.Fatalf("Stack inuse: want %v, got %v", 4<<20, inuse)
	}
}

// Test stack growing in different contexts.
func TestStackGrowth(t *testing.T) {
	switch GOARCH {
	case "386", "arm":
		t.Skipf("skipping test on %q; see issue 8083", GOARCH)
	}
	t.Parallel()
	var wg sync.WaitGroup

	// in a normal goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		growStack()
	}()
	wg.Wait()

	// in locked goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		LockOSThread()
		growStack()
		UnlockOSThread()
	}()
	wg.Wait()

	// in finalizer
	wg.Add(1)
	go func() {
		defer wg.Done()
		done := make(chan bool)
		go func() {
			s := new(string)
			SetFinalizer(s, func(ss *string) {
				growStack()
				done <- true
			})
			s = nil
			done <- true
		}()
		<-done
		GC()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatal("finalizer did not run")
		}
	}()
	wg.Wait()
}

// ... and in init
//func init() {
//	growStack()
//}

func growStack() {
	n := 1 << 10
	if testing.Short() {
		n = 1 << 8
	}
	for i := 0; i < n; i++ {
		x := 0
		growStackIter(&x, i)
		if x != i+1 {
			panic("stack is corrupted")
		}
	}
	GC()
}

// This function is not an anonimous func, so that the compiler can do escape
// analysis and place x on stack (and subsequently stack growth update the pointer).
func growStackIter(p *int, n int) {
	if n == 0 {
		*p = n + 1
		GC()
		return
	}
	*p = n + 1
	x := 0
	growStackIter(&x, n-1)
	if x != n {
		panic("stack is corrupted")
	}
}

func TestStackGrowthCallback(t *testing.T) {
	t.Parallel()
	var wg sync.WaitGroup

	// test stack growth at chan op
	wg.Add(1)
	go func() {
		defer wg.Done()
		c := make(chan int, 1)
		growStackWithCallback(func() {
			c <- 1
			<-c
		})
	}()

	// test stack growth at map op
	wg.Add(1)
	go func() {
		defer wg.Done()
		m := make(map[int]int)
		growStackWithCallback(func() {
			_, _ = m[1]
			m[1] = 1
		})
	}()

	// test stack growth at goroutine creation
	wg.Add(1)
	go func() {
		defer wg.Done()
		growStackWithCallback(func() {
			done := make(chan bool)
			go func() {
				done <- true
			}()
			<-done
		})
	}()

	wg.Wait()
}

func growStackWithCallback(cb func()) {
	var f func(n int)
	f = func(n int) {
		if n == 0 {
			cb()
			return
		}
		f(n - 1)
	}
	for i := 0; i < 1<<10; i++ {
		f(i)
	}
}

// TestDeferPtrs tests the adjustment of Defer's argument pointers (p aka &y)
// during a stack copy.
func set(p *int, x int) {
	*p = x
}
func TestDeferPtrs(t *testing.T) {
	var y int

	defer func() {
		if y != 42 {
			t.Errorf("defer's stack references were not adjusted appropriately")
		}
	}()
	defer set(&y, 42)
	growStack()
}

// use about n KB of stack
func useStack(n int) {
	if n == 0 {
		return
	}
	var b [1024]byte // makes frame about 1KB
	useStack(n - 1 + int(b[99]))
}

func growing(c chan int, done chan struct{}) {
	for n := range c {
		useStack(n)
		done <- struct{}{}
	}
	done <- struct{}{}
}

func TestStackCache(t *testing.T) {
	// Allocate a bunch of goroutines and grow their stacks.
	// Repeat a few times to test the stack cache.
	const (
		R = 4
		G = 200
		S = 5
	)
	for i := 0; i < R; i++ {
		var reqchans [G]chan int
		done := make(chan struct{})
		for j := 0; j < G; j++ {
			reqchans[j] = make(chan int)
			go growing(reqchans[j], done)
		}
		for s := 0; s < S; s++ {
			for j := 0; j < G; j++ {
				reqchans[j] <- 1 << uint(s)
			}
			for j := 0; j < G; j++ {
				<-done
			}
		}
		for j := 0; j < G; j++ {
			close(reqchans[j])
		}
		for j := 0; j < G; j++ {
			<-done
		}
	}
}

func TestStackOutput(t *testing.T) {
	b := make([]byte, 1024)
	stk := string(b[:Stack(b, false)])
	if !strings.HasPrefix(stk, "goroutine ") {
		t.Errorf("Stack (len %d):\n%s", len(stk), stk)
		t.Errorf("Stack output should begin with \"goroutine \"")
	}
}

func TestStackAllOutput(t *testing.T) {
	b := make([]byte, 1024)
	stk := string(b[:Stack(b, true)])
	if !strings.HasPrefix(stk, "goroutine ") {
		t.Errorf("Stack (len %d):\n%s", len(stk), stk)
		t.Errorf("Stack output should begin with \"goroutine \"")
	}
}

func TestStackPanic(t *testing.T) {
	// Test that stack copying copies panics correctly.  This is difficult
	// to test because it is very unlikely that the stack will be copied
	// in the middle of gopanic.  But it can happen.
	// To make this test effective, edit panic.go:gopanic and uncomment
	// the GC() call just before freedefer(d).
	defer func() {
		if x := recover(); x == nil {
			t.Errorf("recover failed")
		}
	}()
	useStack(32)
	panic("test panic")
}
