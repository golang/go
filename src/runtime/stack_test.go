// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"strings"
	"sync"
	"sync/atomic"
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
	// Due to broken stack memory accounting (https://golang.org/issue/7468),
	// StackInuse can decrease during function execution, so we cast the values to int64.
	inuse := int64(s1.StackInuse) - int64(s0.StackInuse)
	t.Logf("Inuse %vMB for stack mem", inuse>>20)
	if inuse > 4<<20 {
		t.Fatalf("Stack inuse: want %v, got %v", 4<<20, inuse)
	}
}

// Test stack growing in different contexts.
func TestStackGrowth(t *testing.T) {
	// Don't make this test parallel as this makes the 20 second
	// timeout unreliable on slow builders. (See issue #19381.)

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
		var started uint32
		go func() {
			s := new(string)
			SetFinalizer(s, func(ss *string) {
				atomic.StoreUint32(&started, 1)
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
			if atomic.LoadUint32(&started) == 0 {
				t.Log("finalizer did not start")
			}
			t.Error("finalizer did not run")
			return
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

// This function is not an anonymous func, so that the compiler can do escape
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

type bigBuf [4 * 1024]byte

// TestDeferPtrsGoexit is like TestDeferPtrs but exercises the possibility that the
// stack grows as part of starting the deferred function. It calls Goexit at various
// stack depths, forcing the deferred function (with >4kB of args) to be run at
// the bottom of the stack. The goal is to find a stack depth less than 4kB from
// the end of the stack. Each trial runs in a different goroutine so that an earlier
// stack growth does not invalidate a later attempt.
func TestDeferPtrsGoexit(t *testing.T) {
	for i := 0; i < 100; i++ {
		c := make(chan int, 1)
		go testDeferPtrsGoexit(c, i)
		if n := <-c; n != 42 {
			t.Fatalf("defer's stack references were not adjusted appropriately (i=%d n=%d)", i, n)
		}
	}
}

func testDeferPtrsGoexit(c chan int, i int) {
	var y int
	defer func() {
		c <- y
	}()
	defer setBig(&y, 42, bigBuf{})
	useStackAndCall(i, Goexit)
}

func setBig(p *int, x int, b bigBuf) {
	*p = x
}

// TestDeferPtrsPanic is like TestDeferPtrsGoexit, but it's using panic instead
// of Goexit to run the Defers. Those two are different execution paths
// in the runtime.
func TestDeferPtrsPanic(t *testing.T) {
	for i := 0; i < 100; i++ {
		c := make(chan int, 1)
		go testDeferPtrsGoexit(c, i)
		if n := <-c; n != 42 {
			t.Fatalf("defer's stack references were not adjusted appropriately (i=%d n=%d)", i, n)
		}
	}
}

func testDeferPtrsPanic(c chan int, i int) {
	var y int
	defer func() {
		if recover() == nil {
			c <- -1
			return
		}
		c <- y
	}()
	defer setBig(&y, 42, bigBuf{})
	useStackAndCall(i, func() { panic(1) })
}

// TestPanicUseStack checks that a chain of Panic structs on the stack are
// updated correctly if the stack grows during the deferred execution that
// happens as a result of the panic.
func TestPanicUseStack(t *testing.T) {
	pc := make([]uintptr, 10000)
	defer func() {
		recover()
		Callers(0, pc) // force stack walk
		useStackAndCall(100, func() {
			defer func() {
				recover()
				Callers(0, pc) // force stack walk
				useStackAndCall(200, func() {
					defer func() {
						recover()
						Callers(0, pc) // force stack walk
					}()
					panic(3)
				})
			}()
			panic(2)
		})
	}()
	panic(1)
}

func TestPanicFar(t *testing.T) {
	var xtree *xtreeNode
	pc := make([]uintptr, 10000)
	defer func() {
		// At this point we created a large stack and unwound
		// it via recovery. Force a stack walk, which will
		// check the stack's consistency.
		Callers(0, pc)
	}()
	defer func() {
		recover()
	}()
	useStackAndCall(100, func() {
		// Kick off the GC and make it do something nontrivial.
		// (This used to force stack barriers to stick around.)
		xtree = makeTree(18)
		// Give the GC time to start scanning stacks.
		time.Sleep(time.Millisecond)
		panic(1)
	})
	_ = xtree
}

type xtreeNode struct {
	l, r *xtreeNode
}

func makeTree(d int) *xtreeNode {
	if d == 0 {
		return new(xtreeNode)
	}
	return &xtreeNode{makeTree(d - 1), makeTree(d - 1)}
}

// use about n KB of stack and call f
func useStackAndCall(n int, f func()) {
	if n == 0 {
		f()
		return
	}
	var b [1024]byte // makes frame about 1KB
	useStackAndCall(n-1+int(b[99]), f)
}

func useStack(n int) {
	useStackAndCall(n, func() {})
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
	// Test that stack copying copies panics correctly. This is difficult
	// to test because it is very unlikely that the stack will be copied
	// in the middle of gopanic. But it can happen.
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

func BenchmarkStackCopy(b *testing.B) {
	c := make(chan bool)
	for i := 0; i < b.N; i++ {
		go func() {
			count(1000000)
			c <- true
		}()
		<-c
	}
}

func count(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count(n-1)
}

func BenchmarkStackCopyNoCache(b *testing.B) {
	c := make(chan bool)
	for i := 0; i < b.N; i++ {
		go func() {
			count1(1000000)
			c <- true
		}()
		<-c
	}
}

func count1(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count2(n-1)
}

func count2(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count3(n-1)
}

func count3(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count4(n-1)
}

func count4(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count5(n-1)
}

func count5(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count6(n-1)
}

func count6(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count7(n-1)
}

func count7(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count8(n-1)
}

func count8(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count9(n-1)
}

func count9(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count10(n-1)
}

func count10(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count11(n-1)
}

func count11(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count12(n-1)
}

func count12(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count13(n-1)
}

func count13(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count14(n-1)
}

func count14(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count15(n-1)
}

func count15(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count16(n-1)
}

func count16(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count17(n-1)
}

func count17(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count18(n-1)
}

func count18(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count19(n-1)
}

func count19(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count20(n-1)
}

func count20(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count21(n-1)
}

func count21(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count22(n-1)
}

func count22(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count23(n-1)
}

func count23(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + count1(n-1)
}
