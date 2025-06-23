// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/testenv"
	"reflect"
	"regexp"
	. "runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	_ "unsafe" // for go:linkname
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
	if *flagQuick {
		t.Skip("-quick")
	}

	var wg sync.WaitGroup

	// in a normal goroutine
	var growDuration time.Duration // For debugging failures
	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now()
		growStack(nil)
		growDuration = time.Since(start)
	}()
	wg.Wait()
	t.Log("first growStack took", growDuration)

	// in locked goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		LockOSThread()
		growStack(nil)
		UnlockOSThread()
	}()
	wg.Wait()

	// in finalizer
	var finalizerStart time.Time
	var started atomic.Bool
	var progress atomic.Uint32
	wg.Add(1)
	s := new(string) // Must be of a type that avoids the tiny allocator, or else the finalizer might not run.
	SetFinalizer(s, func(ss *string) {
		defer wg.Done()
		finalizerStart = time.Now()
		started.Store(true)
		growStack(&progress)
	})
	setFinalizerTime := time.Now()
	s = nil

	if d, ok := t.Deadline(); ok {
		// Pad the timeout by an arbitrary 5% to give the AfterFunc time to run.
		timeout := time.Until(d) * 19 / 20
		timer := time.AfterFunc(timeout, func() {
			// Panic — instead of calling t.Error and returning from the test — so
			// that we get a useful goroutine dump if the test times out, especially
			// if GOTRACEBACK=system or GOTRACEBACK=crash is set.
			if !started.Load() {
				panic("finalizer did not start")
			} else {
				panic(fmt.Sprintf("finalizer started %s ago (%s after registration) and ran %d iterations, but did not return", time.Since(finalizerStart), finalizerStart.Sub(setFinalizerTime), progress.Load()))
			}
		})
		defer timer.Stop()
	}

	GC()
	wg.Wait()
	t.Logf("finalizer started after %s and ran %d iterations in %v", finalizerStart.Sub(setFinalizerTime), progress.Load(), time.Since(finalizerStart))
}

// ... and in init
//func init() {
//	growStack()
//}

func growStack(progress *atomic.Uint32) {
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
		if progress != nil {
			progress.Store(uint32(i))
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
	growStack(nil)
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

//go:noinline
func testDeferLeafSigpanic1() {
	// Cause a sigpanic to be injected in this frame.
	//
	// This function has to be declared before
	// TestDeferLeafSigpanic so the runtime will crash if we think
	// this function's continuation PC is in
	// TestDeferLeafSigpanic.
	*(*int)(nil) = 0
}

// TestDeferLeafSigpanic tests defer matching around leaf functions
// that sigpanic. This is tricky because on LR machines the outer
// function and the inner function have the same SP, but it's critical
// that we match up the defer correctly to get the right liveness map.
// See issue #25499.
func TestDeferLeafSigpanic(t *testing.T) {
	// Push a defer that will walk the stack.
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("expected panic from nil pointer")
		}
		GC()
	}()
	// Call a leaf function. We must set up the exact call stack:
	//
	//  deferring function -> leaf function -> sigpanic
	//
	// On LR machines, the leaf function will have the same SP as
	// the SP pushed for the defer frame.
	testDeferLeafSigpanic1()
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

func BenchmarkStackCopyPtr(b *testing.B) {
	c := make(chan bool)
	for i := 0; i < b.N; i++ {
		go func() {
			i := 1000000
			countp(&i)
			c <- true
		}()
		<-c
	}
}

func countp(n *int) {
	if *n == 0 {
		return
	}
	*n--
	countp(n)
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
	if n <= 0 {
		return 0
	}
	return 1 + count2(n-1)
}

func count2(n int) int  { return 1 + count3(n-1) }
func count3(n int) int  { return 1 + count4(n-1) }
func count4(n int) int  { return 1 + count5(n-1) }
func count5(n int) int  { return 1 + count6(n-1) }
func count6(n int) int  { return 1 + count7(n-1) }
func count7(n int) int  { return 1 + count8(n-1) }
func count8(n int) int  { return 1 + count9(n-1) }
func count9(n int) int  { return 1 + count10(n-1) }
func count10(n int) int { return 1 + count11(n-1) }
func count11(n int) int { return 1 + count12(n-1) }
func count12(n int) int { return 1 + count13(n-1) }
func count13(n int) int { return 1 + count14(n-1) }
func count14(n int) int { return 1 + count15(n-1) }
func count15(n int) int { return 1 + count16(n-1) }
func count16(n int) int { return 1 + count17(n-1) }
func count17(n int) int { return 1 + count18(n-1) }
func count18(n int) int { return 1 + count19(n-1) }
func count19(n int) int { return 1 + count20(n-1) }
func count20(n int) int { return 1 + count21(n-1) }
func count21(n int) int { return 1 + count22(n-1) }
func count22(n int) int { return 1 + count23(n-1) }
func count23(n int) int { return 1 + count1(n-1) }

type stkobjT struct {
	p *stkobjT
	x int64
	y [20]int // consume some stack
}

// Sum creates a linked list of stkobjTs.
func Sum(n int64, p *stkobjT) {
	if n == 0 {
		return
	}
	s := stkobjT{p: p, x: n}
	Sum(n-1, &s)
	p.x += s.x
}

func BenchmarkStackCopyWithStkobj(b *testing.B) {
	c := make(chan bool)
	for i := 0; i < b.N; i++ {
		go func() {
			var s stkobjT
			Sum(100000, &s)
			c <- true
		}()
		<-c
	}
}

func BenchmarkIssue18138(b *testing.B) {
	// Channel with N "can run a goroutine" tokens
	const N = 10
	c := make(chan []byte, N)
	for i := 0; i < N; i++ {
		c <- make([]byte, 1)
	}

	for i := 0; i < b.N; i++ {
		<-c // get token
		go func() {
			useStackPtrs(1000, false) // uses ~1MB max
			m := make([]byte, 8192)   // make GC trigger occasionally
			c <- m                    // return token
		}()
	}
}

func useStackPtrs(n int, b bool) {
	if b {
		// This code contributes to the stack frame size, and hence to the
		// stack copying cost. But since b is always false, it costs no
		// execution time (not even the zeroing of a).
		var a [128]*int // 1KB of pointers
		a[n] = &n
		n = *a[0]
	}
	if n == 0 {
		return
	}
	useStackPtrs(n-1, b)
}

type structWithMethod struct{}

func (s structWithMethod) caller() string {
	_, file, line, ok := Caller(1)
	if !ok {
		panic("Caller failed")
	}
	return fmt.Sprintf("%s:%d", file, line)
}

func (s structWithMethod) callers() []uintptr {
	pc := make([]uintptr, 16)
	return pc[:Callers(0, pc)]
}

func (s structWithMethod) stack() string {
	buf := make([]byte, 4<<10)
	return string(buf[:Stack(buf, false)])
}

func (s structWithMethod) nop() {}

func (s structWithMethod) inlinablePanic() { panic("panic") }

func TestStackWrapperCaller(t *testing.T) {
	var d structWithMethod
	// Force the compiler to construct a wrapper method.
	wrapper := (*structWithMethod).caller
	// Check that the wrapper doesn't affect the stack trace.
	if dc, ic := d.caller(), wrapper(&d); dc != ic {
		t.Fatalf("direct caller %q != indirect caller %q", dc, ic)
	}
}

func TestStackWrapperCallers(t *testing.T) {
	var d structWithMethod
	wrapper := (*structWithMethod).callers
	// Check that <autogenerated> doesn't appear in the stack trace.
	pcs := wrapper(&d)
	frames := CallersFrames(pcs)
	for {
		fr, more := frames.Next()
		if fr.File == "<autogenerated>" {
			t.Fatalf("<autogenerated> appears in stack trace: %+v", fr)
		}
		if !more {
			break
		}
	}
}

func TestStackWrapperStack(t *testing.T) {
	var d structWithMethod
	wrapper := (*structWithMethod).stack
	// Check that <autogenerated> doesn't appear in the stack trace.
	stk := wrapper(&d)
	if strings.Contains(stk, "<autogenerated>") {
		t.Fatalf("<autogenerated> appears in stack trace:\n%s", stk)
	}
}

func TestStackWrapperStackInlinePanic(t *testing.T) {
	// Test that inline unwinding correctly tracks the callee by creating a
	// stack of the form wrapper -> inlined function -> panic. If we mess up
	// callee tracking, it will look like the wrapper called panic and we'll see
	// the wrapper in the stack trace.
	var d structWithMethod
	wrapper := (*structWithMethod).inlinablePanic
	defer func() {
		err := recover()
		if err == nil {
			t.Fatalf("expected panic")
		}
		buf := make([]byte, 4<<10)
		stk := string(buf[:Stack(buf, false)])
		if strings.Contains(stk, "<autogenerated>") {
			t.Fatalf("<autogenerated> appears in stack trace:\n%s", stk)
		}
		// Self-check: make sure inlinablePanic got inlined.
		if !testenv.OptimizationOff() {
			if !strings.Contains(stk, "inlinablePanic(...)") {
				t.Fatalf("inlinablePanic not inlined")
			}
		}
	}()
	wrapper(&d)
}

type I interface {
	M()
}

func TestStackWrapperStackPanic(t *testing.T) {
	t.Run("sigpanic", func(t *testing.T) {
		// nil calls to interface methods cause a sigpanic.
		testStackWrapperPanic(t, func() { I.M(nil) }, "runtime_test.I.M")
	})
	t.Run("panicwrap", func(t *testing.T) {
		// Nil calls to value method wrappers call panicwrap.
		wrapper := (*structWithMethod).nop
		testStackWrapperPanic(t, func() { wrapper(nil) }, "runtime_test.(*structWithMethod).nop")
	})
}

func testStackWrapperPanic(t *testing.T, cb func(), expect string) {
	// Test that the stack trace from a panicking wrapper includes
	// the wrapper, even though elide these when they don't panic.
	t.Run("CallersFrames", func(t *testing.T) {
		defer func() {
			err := recover()
			if err == nil {
				t.Fatalf("expected panic")
			}
			pcs := make([]uintptr, 10)
			n := Callers(0, pcs)
			frames := CallersFrames(pcs[:n])
			for {
				frame, more := frames.Next()
				t.Log(frame.Function)
				if frame.Function == expect {
					return
				}
				if !more {
					break
				}
			}
			t.Fatalf("panicking wrapper %s missing from stack trace", expect)
		}()
		cb()
	})
	t.Run("Stack", func(t *testing.T) {
		defer func() {
			err := recover()
			if err == nil {
				t.Fatalf("expected panic")
			}
			buf := make([]byte, 4<<10)
			stk := string(buf[:Stack(buf, false)])
			if !strings.Contains(stk, "\n"+expect) {
				t.Fatalf("panicking wrapper %s missing from stack trace:\n%s", expect, stk)
			}
		}()
		cb()
	})
}

func TestCallersFromWrapper(t *testing.T) {
	// Test that invoking CallersFrames on a stack where the first
	// PC is an autogenerated wrapper keeps the wrapper in the
	// trace. Normally we elide these, assuming that the wrapper
	// calls the thing you actually wanted to see, but in this
	// case we need to keep it.
	pc := reflect.ValueOf(I.M).Pointer()
	frames := CallersFrames([]uintptr{pc})
	frame, more := frames.Next()
	if frame.Function != "runtime_test.I.M" {
		t.Fatalf("want function %s, got %s", "runtime_test.I.M", frame.Function)
	}
	if more {
		t.Fatalf("want 1 frame, got > 1")
	}
}

func TestTracebackSystemstack(t *testing.T) {
	if GOARCH == "ppc64" || GOARCH == "ppc64le" {
		t.Skip("systemstack tail call not implemented on ppc64x")
	}

	// Test that profiles correctly jump over systemstack,
	// including nested systemstack calls.
	pcs := make([]uintptr, 20)
	pcs = pcs[:TracebackSystemstack(pcs, 5)]
	// Check that runtime.TracebackSystemstack appears five times
	// and that we see TestTracebackSystemstack.
	countIn, countOut := 0, 0
	frames := CallersFrames(pcs)
	var tb strings.Builder
	for {
		frame, more := frames.Next()
		fmt.Fprintf(&tb, "\n%s+0x%x %s:%d", frame.Function, frame.PC-frame.Entry, frame.File, frame.Line)
		switch frame.Function {
		case "runtime.TracebackSystemstack":
			countIn++
		case "runtime_test.TestTracebackSystemstack":
			countOut++
		}
		if !more {
			break
		}
	}
	if countIn != 5 || countOut != 1 {
		t.Fatalf("expected 5 calls to TracebackSystemstack and 1 call to TestTracebackSystemstack, got:%s", tb.String())
	}
}

func TestTracebackAncestors(t *testing.T) {
	goroutineRegex := regexp.MustCompile(`goroutine [0-9]+ \[`)
	for _, tracebackDepth := range []int{0, 1, 5, 50} {
		output := runTestProg(t, "testprog", "TracebackAncestors", fmt.Sprintf("GODEBUG=tracebackancestors=%d", tracebackDepth))

		numGoroutines := 3
		numFrames := 2
		ancestorsExpected := numGoroutines
		if numGoroutines > tracebackDepth {
			ancestorsExpected = tracebackDepth
		}

		matches := goroutineRegex.FindAllStringSubmatch(output, -1)
		if len(matches) != 2 {
			t.Fatalf("want 2 goroutines, got:\n%s", output)
		}

		// Check functions in the traceback.
		fns := []string{"main.recurseThenCallGo", "main.main", "main.printStack", "main.TracebackAncestors"}
		for _, fn := range fns {
			if !strings.Contains(output, "\n"+fn+"(") {
				t.Fatalf("expected %q function in traceback:\n%s", fn, output)
			}
		}

		if want, count := "originating from goroutine", ancestorsExpected; strings.Count(output, want) != count {
			t.Errorf("output does not contain %d instances of %q:\n%s", count, want, output)
		}

		if want, count := "main.recurseThenCallGo(...)", ancestorsExpected*(numFrames+1); strings.Count(output, want) != count {
			t.Errorf("output does not contain %d instances of %q:\n%s", count, want, output)
		}

		if want, count := "main.recurseThenCallGo(0x", 1; strings.Count(output, want) != count {
			t.Errorf("output does not contain %d instances of %q:\n%s", count, want, output)
		}
	}
}

// Test that defer closure is correctly scanned when the stack is scanned.
func TestDeferLiveness(t *testing.T) {
	output := runTestProg(t, "testprog", "DeferLiveness", "GODEBUG=clobberfree=1")
	if output != "" {
		t.Errorf("output:\n%s\n\nwant no output", output)
	}
}

func TestDeferHeapAndStack(t *testing.T) {
	P := 4     // processors
	N := 10000 //iterations
	D := 200   // stack depth

	if testing.Short() {
		P /= 2
		N /= 10
		D /= 10
	}
	c := make(chan bool)
	for p := 0; p < P; p++ {
		go func() {
			for i := 0; i < N; i++ {
				if deferHeapAndStack(D) != 2*D {
					panic("bad result")
				}
			}
			c <- true
		}()
	}
	for p := 0; p < P; p++ {
		<-c
	}
}

// deferHeapAndStack(n) computes 2*n
func deferHeapAndStack(n int) (r int) {
	if n == 0 {
		return 0
	}
	if n%2 == 0 {
		// heap-allocated defers
		for i := 0; i < 2; i++ {
			defer func() {
				r++
			}()
		}
	} else {
		// stack-allocated defers
		defer func() {
			r++
		}()
		defer func() {
			r++
		}()
	}
	r = deferHeapAndStack(n - 1)
	escapeMe(new([1024]byte)) // force some GCs
	return
}

// Pass a value to escapeMe to force it to escape.
var escapeMe = func(x any) {}

func TestFramePointerAdjust(t *testing.T) {
	switch GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("frame pointer is not supported on %s", GOARCH)
	}
	output := runTestProg(t, "testprog", "FramePointerAdjust")
	if output != "" {
		t.Errorf("output:\n%s\n\nwant no output", output)
	}
}

// TestSystemstackFramePointerAdjust is a regression test for issue 59692 that
// ensures that the frame pointer of systemstack is correctly adjusted. See CL
// 489015 for more details.
func TestSystemstackFramePointerAdjust(t *testing.T) {
	growAndShrinkStack(512, [1024]byte{})
}

// growAndShrinkStack grows the stack of the current goroutine in order to
// shrink it again and verify that all frame pointers on the new stack have
// been correctly adjusted. stackBallast is used to ensure we're not depending
// on the current heuristics of stack shrinking too much.
func growAndShrinkStack(n int, stackBallast [1024]byte) {
	if n <= 0 {
		return
	}
	growAndShrinkStack(n-1, stackBallast)
	ShrinkStackAndVerifyFramePointers()
}
