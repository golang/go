// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package jit_test

import (
	"runtime"
	"runtime/jit"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestMetadataReclaimedAfterUnregister(t *testing.T) {
	const regions = 192
	const rounds = 12
	const base = uintptr(0x51000000)
	const stride = uintptr(1 << 20)
	maps := make([]jit.StackMap, 32)
	for i := range maps {
		maps[i] = jit.StackMap{PCOffset: uintptr(i * 8), FrameWords: 16, PointerMask: []byte{1, 0}}
	}
	// Warm runtime bookkeeping before measuring retained allocation.
	h := jit.Register(jit.Region{Start: base, End: base + stride, Unwind: jit.UnwindStop})
	h.Unregister()
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for round := 0; round < rounds; round++ {
		handles := make([]jit.Handle, regions)
		for i := range handles {
			start := base + uintptr(i)*stride
			handles[i] = jit.Register(jit.Region{Start: start, End: start + stride, Unwind: jit.UnwindStop, StackMaps: maps})
		}
		// Use a non-monotonic deletion order to exercise both tree rotations.
		for i := range handles {
			handles[(i*113)%regions].Unregister()
		}
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	const allowance = uint64(4 << 20)
	if after.OtherSys > before.OtherSys+allowance {
		t.Fatalf("unregistered metadata retained %d OtherSys bytes", after.OtherSys-before.OtherSys)
	}
	if after.HeapAlloc > before.HeapAlloc+allowance {
		t.Fatalf("unregistered metadata retained %d heap bytes", after.HeapAlloc-before.HeapAlloc)
	}
}

//go:noinline
func registerMetadataOwner(done *atomic.Bool) jit.Handle {
	type owner struct {
		value   int
		padding [64]byte
	}
	value := &owner{value: 42}
	runtime.AddCleanup(value, func(flag *atomic.Bool) { flag.Store(true) }, done)
	return jit.Register(jit.Region{
		Start: 0x62000000, End: 0x62001000, Unwind: jit.UnwindDeclare,
		Describe: func(uintptr) (string, string, int, bool) {
			return "metadata-owner", "metadata.go", value.value, true
		},
	})
}

func TestMetadataOwnsDescribeUntilUnregister(t *testing.T) {
	done := new(atomic.Bool)
	h := registerMetadataOwner(done)
	for i := 0; i < 3; i++ {
		runtime.GC()
		frame, _ := runtime.CallersFrames([]uintptr{0x62000001}).Next()
		if frame.Function != "metadata-owner" || frame.Line != 42 || done.Load() {
			t.Fatalf("active metadata lost its Describe owner: %+v, cleaned=%v", frame, done.Load())
		}
	}
	h.Unregister()
	deadline := time.Now().Add(5 * time.Second)
	for !done.Load() && time.Now().Before(deadline) {
		runtime.GC()
		time.Sleep(time.Millisecond)
	}
	if !done.Load() {
		t.Fatal("unregistered Describe owner remained reachable")
	}
}

func TestMetadataLookupConcurrentWithUpdatesAndGC(t *testing.T) {
	const base = uintptr(0x63000000)
	anchor := jit.Register(jit.Region{
		Start: base, End: base + 4096, Unwind: jit.UnwindDeclare,
		Describe: func(uintptr) (string, string, int, bool) { return "anchor", "anchor.go", 7, true },
	})
	defer anchor.Unregister()
	var workers sync.WaitGroup
	for reader := 0; reader < 4; reader++ {
		workers.Go(func() {
			for i := 0; i < 500; i++ {
				frame, _ := runtime.CallersFrames([]uintptr{base + 1}).Next()
				if frame.Function != "anchor" || frame.Line != 7 {
					t.Errorf("lookup during registry update = %+v", frame)
					return
				}
			}
		})
	}
	workers.Go(func() {
		for i := 0; i < 10; i++ {
			runtime.GC()
		}
	})
	for i := 0; i < 500; i++ {
		start := base + uintptr(i+1)*8192
		h := jit.Register(jit.Region{Start: start, End: start + 4096, Unwind: jit.UnwindStop})
		h.Unregister()
	}
	workers.Wait()
}
