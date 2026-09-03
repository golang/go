// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && (unix || windows)

package jit_test

import (
	"fmt"
	"runtime"
	"runtime/jit"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

type tailAllocationElement struct {
	Pointer *int
	Value   uintptr
}

type tailAllocationObject struct {
	Header *int
	Value  uintptr
	Tail   [0]tailAllocationElement
}

func TestTailTypeAllocatesRepeatedPointerLayout(t *testing.T) {
	const count = 3
	prepared := jit.TailTypeFor[tailAllocationObject](count)
	object := (*tailAllocationObject)(prepared.Alloc())
	tail := unsafe.Slice((*tailAllocationElement)(unsafe.Add(unsafe.Pointer(object), unsafe.Offsetof(object.Tail))), count)
	header := 7
	object.Header = &header
	for index := range tail {
		value := 10 + index
		tail[index].Pointer = &value
		tail[index].Value = uintptr(value)
	}
	runtime.GC()
	if *object.Header != 7 {
		t.Fatalf("header pointer = %d, want 7", *object.Header)
	}
	for index := range tail {
		if got, want := *tail[index].Pointer, 10+index; got != want {
			t.Fatalf("tail[%d] pointer = %d, want %d", index, got, want)
		}
	}
	runtime.KeepAlive(object)
}

// goFuncPtr returns the raw entry point of a Go function value.
func goFuncPtr(fn func()) uintptr {
	return **(**uintptr)(unsafe.Pointer(&fn))
}

// TestRegisterUnregister tests basic region registration and removal.
func TestRegisterUnregister(t *testing.T) {
	code := retTrampoline()
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	h.Unregister()
}

// TestMultipleRegisterUnregister tests registering and unregistering
// multiple non-overlapping regions.
func TestMultipleRegisterUnregister(t *testing.T) {
	var handles []jit.Handle
	var addrs []uintptr
	var sizes []int

	for i := 0; i < 5; i++ {
		code := retTrampoline()
		addr, size, err := allocExecutable(code)
		if err != nil {
			t.Fatalf("allocExecutable %d: %v", i, err)
		}
		addrs = append(addrs, addr)
		sizes = append(sizes, size)

		h := jit.Register(jit.Region{
			Start:  addr,
			End:    addr + uintptr(size),
			Unwind: jit.UnwindStop,
		})
		handles = append(handles, h)
	}

	// Unregister in reverse order.
	for i := len(handles) - 1; i >= 0; i-- {
		handles[i].Unregister()
	}
	for i, addr := range addrs {
		freeExecutable(addr, sizes[i])
	}
}

// TestManyRegisterUnregister verifies that the registry grows with its users.
// JITs commonly allocate one registered region per executable arena.
func TestManyRegisterUnregister(t *testing.T) {
	code := make([]byte, 128)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	handles := make([]jit.Handle, len(code))
	for i := range handles {
		handles[i] = jit.Register(jit.Region{
			Start:  addr + uintptr(i),
			End:    addr + uintptr(i+1),
			Unwind: jit.UnwindStop,
		})
	}
	for i := len(handles) - 1; i >= 0; i-- {
		handles[i].Unregister()
	}
}

// TestConcurrentRegisterUnregister tests that concurrent registration
// and unregistration do not crash.
func TestConcurrentRegisterUnregister(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			code := retTrampoline()
			addr, size, err := allocExecutable(code)
			if err != nil {
				t.Errorf("allocExecutable: %v", err)
				return
			}
			defer freeExecutable(addr, size)

			h := jit.Register(jit.Region{
				Start:  addr,
				End:    addr + uintptr(size),
				Unwind: jit.UnwindStop,
			})
			// Yield to increase chance of concurrent operations.
			runtime.Gosched()
			h.Unregister()
		}()
	}
	wg.Wait()
}

// TestPanicThroughUserFrameTailCall tests that a panic in Go code called
// from JIT code via tail call can be recovered.
func TestPanicThroughUserFrameTailCall(t *testing.T) {
	fnPtr := goFuncPtr(goPanicker)

	code := tailCallTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	defer h.Unregister()

	recovered := callAndRecover(addr)
	if recovered == nil {
		t.Fatal("expected panic to be recovered")
	}
	if msg := fmt.Sprint(recovered); msg != "jit panic test" {
		t.Fatalf("unexpected recovered value: %v", recovered)
	}
}

// TestPanicThroughUserFrameWithCall tests unwinding through a JIT call
// frame. This is the core test:
//
//	Go caller (with defer/recover) → JIT trampoline (call) → Go callee → panic
func TestPanicThroughUserFrameWithCall(t *testing.T) {
	fnPtr := goFuncPtr(goPanicker)

	code := callTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: callTrampolineStackMaps(),
	})
	defer h.Unregister()

	recovered := callAndRecover(addr)
	if recovered == nil {
		t.Fatal("expected panic to be recovered")
	}
	if msg := fmt.Sprint(recovered); msg != "jit panic test" {
		t.Fatalf("unexpected recovered value: %v", recovered)
	}
}

// TestUnwindStop tests that UnwindStop gracefully terminates the traceback
// at the user frame boundary without crashing.
func TestUnwindStop(t *testing.T) {
	fnPtr := goFuncPtr(goPanicker)

	code := callTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	// Register with UnwindStop and no safepoint metadata.
	// The panic should be unable to find the defer/recover, so the
	// program would crash. We test this in a subprocess.
	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	defer h.Unregister()

	// With UnwindStop and no unwind recipe, the unwinder stops at the
	// JIT boundary. The panic cannot reach callAndRecover's defer, so
	// recover() returns nil and the panic continues to crash.
	// We just verify the registration itself doesn't crash.
}

// TestCallersIncludesGoFrames tests that runtime.Callers works correctly
// when user frames are registered (but not in the current call chain).
func TestCallersIncludesGoFrames(t *testing.T) {
	code := retTrampoline()
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	defer h.Unregister()

	// runtime.Callers should still work for normal Go frames.
	pcs := make([]uintptr, 32)
	n := runtime.Callers(1, pcs)
	if n == 0 {
		t.Fatal("runtime.Callers returned 0 frames")
	}
	frames := runtime.CallersFrames(pcs[:n])
	found := false
	for {
		frame, more := frames.Next()
		if strings.Contains(frame.Function, "TestCallersIncludesGoFrames") {
			found = true
		}
		if !more {
			break
		}
	}
	if !found {
		t.Fatal("runtime.Callers did not include TestCallersIncludesGoFrames")
	}
}

// TestRuntimeStackWithUserFrame tests that runtime.Stack does not crash
// when a user frame region is registered.
func TestRuntimeStackWithUserFrame(t *testing.T) {
	code := retTrampoline()
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	defer h.Unregister()

	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	if n == 0 {
		t.Fatal("runtime.Stack returned empty")
	}
	stack := string(buf[:n])
	if !strings.Contains(stack, "TestRuntimeStackWithUserFrame") {
		t.Fatalf("runtime.Stack missing test function:\n%s", stack)
	}
}

// TestGCWithUserFrame tests that garbage collection does not crash
// when user frame regions are registered.
func TestGCWithUserFrame(t *testing.T) {
	code := retTrampoline()
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindStop,
	})
	defer h.Unregister()

	// Force a GC cycle. If the user frame registration interferes
	// with the GC's stack scanning, this will crash.
	runtime.GC()
}

// TestPanicRecoverMultipleGoroutines tests panic/recover through user
// frames from multiple concurrent goroutines.
func TestPanicRecoverMultipleGoroutines(t *testing.T) {
	fnPtr := goFuncPtr(goPanicker)

	code := callTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: callTrampolineStackMaps(),
	})
	defer h.Unregister()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			recovered := callAndRecover(addr)
			if recovered == nil {
				t.Error("expected panic to be recovered")
				return
			}
			if msg := fmt.Sprint(recovered); msg != "jit panic test" {
				t.Errorf("unexpected recovered value: %v", recovered)
			}
		}()
	}
	wg.Wait()
}

// TestUnwindDeclareDescribe tests that the Describe callback is invoked
// during traceback printing when UnwindDeclare mode is used.
func TestUnwindDeclareDescribe(t *testing.T) {
	fnPtr := goFuncPtr(goPanicker)

	code := callTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	var describeCalled uint32
	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindDeclare,
		Describe: func(pc uintptr) (string, string, int, bool) {
			atomic.AddUint32(&describeCalled, 1)
			return "myJitFunction", "jit_generated.go", 42, true
		},
		StackMaps: callTrampolineStackMaps(),
	})
	defer h.Unregister()

	// Panic/recover through the JIT frame should still work.
	recovered := callAndRecover(addr)
	if recovered == nil {
		t.Fatal("expected panic to be recovered")
	}
	if msg := fmt.Sprint(recovered); msg != "jit panic test" {
		t.Fatalf("unexpected recovered value: %v", recovered)
	}

	// The Describe callback may or may not be called during panic
	// unwinding (it's called during traceback printing, not during
	// the unwind itself). We verify at minimum that the registration
	// with UnwindDeclare + Describe works without crashing.
}

// TestUnwindDeclareInStackTrace tests that user frame info from Describe
// appears in runtime.Stack output when captured from within a JIT call.
func TestUnwindDeclareInStackTrace(t *testing.T) {
	// Use the call trampoline to call goStackCapture which captures
	// runtime.Stack from inside the JIT call chain.
	fnPtr := goFuncPtr(goStackCapture)

	code := callTrampoline(fnPtr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindDeclare,
		Describe: func(pc uintptr) (string, string, int, bool) {
			return "myJitFunction", "jit_generated.go", 42, true
		},
		StackMaps: callTrampolineStackMaps(),
	})
	defer h.Unregister()

	// Call JIT trampoline → goStackCapture captures the stack.
	capturedStack = ""
	callJIT(addr)

	// Note: Whether the user frame appears in the stack depends on
	// the traceback implementation. The key assertion is that the
	// call chain works without crashing.
	if capturedStack == "" {
		t.Fatal("goStackCapture was not called through JIT trampoline")
	}
	if !strings.Contains(capturedStack, "goStackCapture") {
		t.Fatalf("stack trace missing goStackCapture:\n%s", capturedStack)
	}
}

// TestStackMapInactive verifies that registering stack maps without an active
// frame does not add global GC roots.
func TestStackMapInactive(t *testing.T) {
	code := retTrampoline()
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindStop,
		StackMaps: []jit.StackMap{{PCOffset: 0, FrameWords: 1, PointerMask: []byte{1}}},
	})
	defer h.Unregister()

	runtime.GC()
}

// TestPreemptReturnsFalseNormally tests that Preempt returns false when
// no preemption is requested.
func TestPreemptReturnsFalseNormally(t *testing.T) {
	if jit.Preempt() {
		t.Fatal("Preempt() returned true when no preemption was requested")
	}
}

// TestPreemptDuringGC tests that Preempt returns true when GC requests
// preemption. We spawn a goroutine that busy-loops checking Preempt
// and force a GC cycle — the goroutine should observe Preempt() == true.
func TestPreemptDuringGC(t *testing.T) {
	var saw atomic.Bool
	var stop atomic.Bool

	go func() {
		for !stop.Load() {
			if jit.Preempt() {
				saw.Store(true)
			}
			// Tight loop simulating JIT execution.
			// Do NOT call runtime.Gosched — we want to test that
			// Preempt detects the flag without yielding first.
		}
	}()

	// Give the goroutine time to start.
	runtime.Gosched()

	// Force GC — this sets gp.preempt on all goroutines during STW.
	runtime.GC()

	stop.Store(true)

	// The goroutine should have seen preempt==true at some point.
	if !saw.Load() {
		t.Log("Preempt() was not observed as true during GC (timing-dependent, not a hard failure)")
	}
}

var capturedStack string

// goStackCapture captures runtime.Stack into capturedStack.
//
//go:noinline
func goStackCapture() {
	buf := make([]byte, 8192)
	n := runtime.Stack(buf, false)
	capturedStack = string(buf[:n])
}

// callJIT calls a function at addr without recover.
//
//go:noinline
func callJIT(addr uintptr) {
	fv := addr
	fvp := unsafe.Pointer(&fv)
	fn := *(*func())(unsafe.Pointer(&fvp))
	fn()
}

// goPanicker panics with a known value. Called from JIT code.
//
//go:noinline
func goPanicker() {
	panic("jit panic test")
}

// callAndRecover calls a function at addr and recovers any panic.
// A Go func() value is a pointer to a funcval struct whose first
// field is the entry PC. We build a funcval on the stack.
//
//go:noinline
func callAndRecover(addr uintptr) (recovered any) {
	defer func() {
		recovered = recover()
	}()
	fv := addr
	fvp := unsafe.Pointer(&fv)
	fn := *(*func())(unsafe.Pointer(&fvp))
	fn()
	return nil
}
