// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// the race detector does not like our pointer shenanigans
// while checking the stack.

//go:build goexperiment.runtimesecret && (arm64 || amd64) && linux && !race

package secret

import (
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"
)

type secretType int64

const secretValue = 0x53c237_53c237

// S is a type that might have some secrets in it.
type S [100]secretType

// makeS makes an S with secrets in it.
//
//go:noinline
func makeS() S {
	// Note: noinline ensures this doesn't get inlined and
	// completely optimized away.
	var s S
	for i := range s {
		s[i] = secretValue
	}
	return s
}

// heapS allocates an S on the heap with secrets in it.
//
//go:noinline
func heapS() *S {
	// Note: noinline forces heap allocation
	s := makeS()
	return &s
}

// for the tiny allocator
//
//go:noinline
func heapSTiny() *secretType {
	s := new(secretType(secretValue))
	return s
}

// Test that when we allocate inside secret.Do, the resulting
// allocations are zeroed by the garbage collector when they
// are freed.
// See runtime/mheap.go:freeSpecial.
func TestHeap(t *testing.T) {
	var u uintptr
	Do(func() {
		u = uintptr(unsafe.Pointer(heapS()))
	})

	runtime.GC()

	// Check that object got zeroed.
	checkRangeForSecret(t, u, u+unsafe.Sizeof(S{}))
	// Also check our stack, just because we can.
	checkStackForSecret(t)
}

func TestHeapTiny(t *testing.T) {
	var u uintptr
	Do(func() {
		u = uintptr(unsafe.Pointer(heapSTiny()))
	})
	runtime.GC()

	// Check that object got zeroed.
	checkRangeForSecret(t, u, u+unsafe.Sizeof(secretType(0)))
	// Also check our stack, just because we can.
	checkStackForSecret(t)
}

// Test that when we return from secret.Do, we zero the stack used
// by the argument to secret.Do.
// See runtime/secret.go:secret_dec.
func TestStack(t *testing.T) {
	checkStackForSecret(t) // if this fails, something is wrong with the test

	Do(func() {
		s := makeS()
		use(&s)
	})

	checkStackForSecret(t)
}

//go:noinline
func use(s *S) {
	// Note: noinline prevents dead variable elimination.
}

// Test that when we copy a stack, we zero the old one.
// See runtime/stack.go:copystack.
func TestStackCopy(t *testing.T) {
	checkStackForSecret(t) // if this fails, something is wrong with the test

	var lo, hi uintptr
	Do(func() {
		// Put some secrets on the current stack frame.
		s := makeS()
		use(&s)
		// Remember the current stack.
		lo, hi = getStack()
		// Use a lot more stack to force a stack copy.
		growStack()
	})
	checkRangeForSecret(t, lo, hi) // pre-grow stack
	checkStackForSecret(t)         // post-grow stack (just because we can)
}

func growStack() {
	growStack1(1000)
}
func growStack1(n int) {
	if n == 0 {
		return
	}
	growStack1(n - 1)
}

func TestPanic(t *testing.T) {
	checkStackForSecret(t) // if this fails, something is wrong with the test

	defer func() {
		checkStackForSecret(t)

		p := recover()
		if p == nil {
			t.Errorf("panic squashed")
			return
		}
		var e error
		var ok bool
		if e, ok = p.(error); !ok {
			t.Errorf("panic not an error")
		}
		if !strings.Contains(e.Error(), "divide by zero") {
			t.Errorf("panic not a divide by zero error: %s", e.Error())
		}
		var pcs [10]uintptr
		n := runtime.Callers(0, pcs[:])
		frames := runtime.CallersFrames(pcs[:n])
		for {
			frame, more := frames.Next()
			if strings.Contains(frame.Function, "dividePanic") {
				t.Errorf("secret function in traceback")
			}
			if !more {
				break
			}
		}
	}()
	Do(dividePanic)
}

func dividePanic() {
	s := makeS()
	use(&s)
	_ = 8 / zero
}

var zero int

func TestGoExit(t *testing.T) {
	checkStackForSecret(t) // if this fails, something is wrong with the test

	c := make(chan uintptr, 2)

	go func() {
		// Run the test in a separate goroutine
		defer func() {
			// Tell original goroutine what our stack is
			// so it can check it for secrets.
			lo, hi := getStack()
			c <- lo
			c <- hi
		}()
		Do(func() {
			s := makeS()
			use(&s)
			// there's an entire round-trip through the scheduler between here
			// and when we are able to check if the registers are still dirtied, and we're
			// not guaranteed to run on the same M. Make a best effort attempt anyway
			loadRegisters(unsafe.Pointer(&s))
			runtime.Goexit()
		})
		t.Errorf("goexit didn't happen")
	}()
	lo := <-c
	hi := <-c
	// We want to wait until the other goroutine has finished Goexiting and
	// cleared its stack. There's no signal for that, so just wait a bit.
	time.Sleep(1 * time.Millisecond)

	checkRangeForSecret(t, lo, hi)

	var spillArea [64]secretType
	n := spillRegisters(unsafe.Pointer(&spillArea))
	if n > unsafe.Sizeof(spillArea) {
		t.Fatalf("spill area overrun %d\n", n)
	}
	for i, v := range spillArea {
		if v == secretValue {
			t.Errorf("secret found in spill slot %d", i)
		}
	}
}

func checkStackForSecret(t *testing.T) {
	t.Helper()
	lo, hi := getStack()
	checkRangeForSecret(t, lo, hi)
}
func checkRangeForSecret(t *testing.T, lo, hi uintptr) {
	t.Helper()
	for p := lo; p < hi; p += unsafe.Sizeof(secretType(0)) {
		v := *(*secretType)(unsafe.Pointer(p))
		if v == secretValue {
			t.Errorf("secret found in [%x,%x] at %x", lo, hi, p)
		}
	}
}

func TestRegisters(t *testing.T) {
	Do(func() {
		s := makeS()
		loadRegisters(unsafe.Pointer(&s))
	})
	var spillArea [64]secretType
	n := spillRegisters(unsafe.Pointer(&spillArea))
	if n > unsafe.Sizeof(spillArea) {
		t.Fatalf("spill area overrun %d\n", n)
	}
	for i, v := range spillArea {
		if v == secretValue {
			t.Errorf("secret found in spill slot %d", i)
		}
	}
}

func TestSignalStacks(t *testing.T) {
	Do(func() {
		s := makeS()
		loadRegisters(unsafe.Pointer(&s))
		// cause a signal with our secret state to dirty
		// at least one of the signal stacks
		func() {
			defer func() {
				x := recover()
				if x == nil {
					panic("did not get panic")
				}
			}()
			var p *int
			*p = 20
		}()
	})
	// signal stacks aren't cleared until after
	// the next GC after secret.Do returns
	runtime.GC()
	stk := make([]stack, 0, 100)
	stk = appendSignalStacks(stk)
	for _, s := range stk {
		checkRangeForSecret(t, s.lo, s.hi)
	}
}

// hooks into the runtime
func getStack() (uintptr, uintptr)

// Stack is a copy of runtime.stack for testing export.
// Fields must match.
type stack struct {
	lo uintptr
	hi uintptr
}

func appendSignalStacks([]stack) []stack
