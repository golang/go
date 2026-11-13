// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit && amd64 && (unix || windows)

package jit_test

import (
	"runtime"
	"runtime/jit"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

var jitObjectFinalized atomic.Bool

type jitHeapObject struct {
	value   uintptr
	padding [1024]byte
}

// TestStackMapActiveFrame verifies that a GC can walk an active JIT frame
// using the registered return-PC metadata.
func TestStackMapActiveFrame(t *testing.T) {
	code := pointerSlotCallTrampoline(goFuncPtr(forceJITGC), 0)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer h.Unregister()

	callJIT(addr)
}

// TestPointerMapKeepsHeapObjectLive leaves the only typed reference to an
// object in a JIT stack slot. GCs executed by the Go callback must retain it;
// once the JIT frame returns, the finalizer must be able to run.
func TestPointerMapKeepsHeapObjectLive(t *testing.T) {
	jitObjectFinalized.Store(false)
	ptr := newFinalizedJITObject()
	code := pointerSlotCallTrampoline(goFuncPtr(forceJITGC), ptr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindSkip,
	})
	defer h.Unregister()
	maps := growingPointerSlotStackMaps()
	h.AddStackMaps(maps[:10]...)
	h.AddStackMaps(maps[10:]...)
	// AddStackMaps copies the bitmap before publishing it.
	maps[len(maps)-1].PointerMask[0] = 0

	callJIT(addr)
	if jitObjectFinalized.Load() {
		t.Fatal("object referenced by JIT frame was finalized during the call")
	}

	for i := 0; i < 20 && !jitObjectFinalized.Load(); i++ {
		runtime.GC()
		runtime.Gosched()
	}
	if !jitObjectFinalized.Load() {
		t.Fatal("object was still retained after its JIT frame returned")
	}
}

// TestPointerMapKeepsHeapObjectLiveAcrossNestedFrames leaves the only typed
// reference in the outer of two consecutive JIT frames. A GC called through
// the inner frame must scan both registered maps.
func TestPointerMapKeepsHeapObjectLiveAcrossNestedFrames(t *testing.T) {
	jitObjectFinalized.Store(false)
	ptr := newFinalizedJITObject()

	innerCode := pointerSlotCallTrampoline(goFuncPtr(forceJITGC), 0)
	innerAddr, innerSize, err := allocExecutable(innerCode)
	if err != nil {
		t.Fatalf("allocExecutable inner: %v", err)
	}
	defer freeExecutable(innerAddr, innerSize)
	innerHandle := jit.Register(jit.Region{
		Start:     innerAddr,
		End:       innerAddr + uintptr(innerSize),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer innerHandle.Unregister()

	outerCode := pointerSlotCallTrampoline(innerAddr, ptr)
	outerAddr, outerSize, err := allocExecutable(outerCode)
	if err != nil {
		t.Fatalf("allocExecutable outer: %v", err)
	}
	defer freeExecutable(outerAddr, outerSize)
	outerHandle := jit.Register(jit.Region{
		Start:     outerAddr,
		End:       outerAddr + uintptr(outerSize),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer outerHandle.Unregister()

	callJIT(outerAddr)
	if jitObjectFinalized.Load() {
		t.Fatal("object referenced by outer JIT frame was finalized during nested call")
	}

	for i := 0; i < 20 && !jitObjectFinalized.Load(); i++ {
		runtime.GC()
		runtime.Gosched()
	}
	if !jitObjectFinalized.Load() {
		t.Fatal("object was still retained after nested JIT frames returned")
	}
}

// TestStackMapParallelActiveFrames exercises concurrent stack-root jobs. Each
// worker resolves immutable metadata for the frame it is currently walking.
func TestStackMapParallelActiveFrames(t *testing.T) {
	code := pointerSlotCallTrampoline(goFuncPtr(forceJITGC), 0)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer h.Unregister()

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			callJIT(addr)
		}()
	}
	wg.Wait()
}

// TestStackMapPublicationConcurrentWithGC grows a region's append-only map
// table while GCs repeatedly scan an already published active frame. Readers
// may observe either backing array, but must never observe an entry or bitmap
// before it has been initialized completely.
func TestStackMapPublicationConcurrentWithGC(t *testing.T) {
	finalized := new(atomic.Bool)
	ptr := newFinalizedJITObjectFor(finalized)
	code := pointerSlotCallTrampoline(goFuncPtr(forceJITGC), ptr)
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer h.Unregister()

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			callJIT(addr)
		}()
	}

	actual := pointerSlotStackMaps()[0]
	for pc := actual.PCOffset + 1; pc < uintptr(size) && pc < actual.PCOffset+2048; pc++ {
		h.AddStackMaps(jit.StackMap{
			PCOffset:         pc,
			HasUnwind:        true,
			UnwindBaseOffset: actual.UnwindBaseOffset,
			CallerPCOffset:   actual.CallerPCOffset,
			CallerSPOffset:   actual.CallerSPOffset,
		})
	}
	wg.Wait()
	if finalized.Load() {
		t.Fatal("object referenced by an active frame was lost during stack-map publication")
	}
}

// TestPointerMapStackGrowth verifies that copystack uses the JIT pointer map.
// The generated frame stores a pointer to its own second local word, calls Go
// deeply enough to grow the goroutine stack, and then checks that the pointer
// was relocated to the copied frame.
func TestPointerMapStackGrowth(t *testing.T) {
	code := stackPointerCallTrampoline(goFuncPtr(growJITStack))
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:     addr,
		End:       addr + uintptr(size),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer h.Unregister()

	if got := callJITUintptr(addr); got != 1 {
		t.Fatal("pointer in JIT frame was not relocated during stack growth")
	}
}

// TestPointerMapStackGrowthAcrossNestedFrames verifies that stack pointers in
// an outer JIT frame are relocated while stack growth is triggered through a
// directly called inner JIT frame.
func TestPointerMapStackGrowthAcrossNestedFrames(t *testing.T) {
	innerCode := pointerSlotCallTrampoline(goFuncPtr(growJITStack), 0)
	innerAddr, innerSize, err := allocExecutable(innerCode)
	if err != nil {
		t.Fatalf("allocExecutable inner: %v", err)
	}
	defer freeExecutable(innerAddr, innerSize)
	innerHandle := jit.Register(jit.Region{
		Start:     innerAddr,
		End:       innerAddr + uintptr(innerSize),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer innerHandle.Unregister()

	outerCode := stackPointerCallTrampoline(innerAddr)
	outerAddr, outerSize, err := allocExecutable(outerCode)
	if err != nil {
		t.Fatalf("allocExecutable outer: %v", err)
	}
	defer freeExecutable(outerAddr, outerSize)
	outerHandle := jit.Register(jit.Region{
		Start:     outerAddr,
		End:       outerAddr + uintptr(outerSize),
		Unwind:    jit.UnwindSkip,
		StackMaps: pointerSlotStackMaps(),
	})
	defer outerHandle.Unregister()

	if got := callJITUintptr(outerAddr); got != 1 {
		t.Fatal("pointer in outer JIT frame was not relocated during nested stack growth")
	}
}

// TestCallerFramePointerStackGrowth verifies that caller-BP relocation is
// driven by unwind metadata rather than by pretending that the saved frame
// pointer is a GC heap root in PointerMask.
func TestCallerFramePointerStackGrowth(t *testing.T) {
	code := callerFramePointerCallTrampoline(goFuncPtr(growJITStack))
	addr, size, err := allocExecutable(code)
	if err != nil {
		t.Fatalf("allocExecutable: %v", err)
	}
	defer freeExecutable(addr, size)

	h := jit.Register(jit.Region{
		Start:  addr,
		End:    addr + uintptr(size),
		Unwind: jit.UnwindSkip,
		StackMaps: []jit.StackMap{{
			PCOffset:         31,
			FrameWords:       3,
			PointerMask:      []byte{0},
			HasUnwind:        true,
			UnwindBaseOffset: 16,
			CallerPCOffset:   8,
			CallerSPOffset:   16,
			CallerBPOffset:   0,
		}},
	})
	defer h.Unregister()

	if got := callJITUintptr(addr); got != 1 {
		t.Fatal("saved caller frame pointer was not relocated during stack growth")
	}
}

func pointerSlotStackMaps() []jit.StackMap {
	// pointerSlotCallTrampoline's call ends at byte offset 34. The first of
	// its two local words is a pointer at that safepoint.
	return []jit.StackMap{{
		PCOffset:         34,
		FrameWords:       2,
		PointerMask:      []byte{1},
		HasUnwind:        true,
		UnwindBaseOffset: 16,
		CallerPCOffset:   8,
		CallerSPOffset:   16,
	}}
}

func growingPointerSlotStackMaps() []jit.StackMap {
	actual := pointerSlotStackMaps()[0]
	maps := make([]jit.StackMap, actual.PCOffset)
	for i := range maps {
		maps[i] = jit.StackMap{
			PCOffset:         uintptr(i + 1),
			HasUnwind:        true,
			UnwindBaseOffset: actual.UnwindBaseOffset,
			CallerPCOffset:   actual.CallerPCOffset,
			CallerSPOffset:   actual.CallerSPOffset,
		}
	}
	maps[len(maps)-1] = actual
	return maps
}

//go:noinline
func forceJITGC() {
	for i := 0; i < 3; i++ {
		runtime.GC()
		runtime.Gosched()
	}
}

//go:noinline
func newFinalizedJITObject() uintptr {
	return newFinalizedJITObjectFor(&jitObjectFinalized)
}

//go:noinline
func newFinalizedJITObjectFor(finalized *atomic.Bool) uintptr {
	obj := &jitHeapObject{value: 0x12345678}
	runtime.SetFinalizer(obj, func(*jitHeapObject) { finalized.Store(true) })
	return uintptr(unsafe.Pointer(obj))
}

//go:noinline
func growJITStack() {
	growJITStackBy(128)
}

//go:noinline
func growJITStackBy(depth int) uintptr {
	var space [1024]byte
	space[0] = byte(depth)
	if depth == 0 {
		return uintptr(space[0])
	}
	return uintptr(space[0]) + growJITStackBy(depth-1)
}

//go:noinline
func callJITUintptr(addr uintptr) uintptr {
	fv := addr
	fvp := unsafe.Pointer(&fv)
	fn := *(*func() uintptr)(unsafe.Pointer(&fvp))
	return fn()
}
