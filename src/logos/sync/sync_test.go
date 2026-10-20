// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Attack tests for M8-S5: logos/sync Locked[T] panic-safe unlock (#176),
// Hot[T] concurrent-update safety (#26), retire-free updates (#153), and
// the atomic.Pointer-only layout (#154). Benchmarks are report-only
// (T-PERF): BenchmarkWithLock targets within 2 ns/op of manual
// Lock/defer Unlock, and BenchmarkHotLoad (hot_load_bench_test.go) reports
// Hot.Load against a plain atomic.Pointer load.
package sync_test

import (
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"runtime"

	logosync "logos/sync"
)

type lockConfig struct {
	Port int
	Name string
}

func TestLockedWithLockMutates(t *testing.T) {
	var l logosync.Locked[lockConfig]
	l.WithLock(func(c *lockConfig) {
		c.Port = 8080
		c.Name = "svc"
	})
	l.WithLock(func(c *lockConfig) {
		if c.Port != 8080 || c.Name != "svc" {
			t.Fatalf("WithLock did not persist mutation: %+v", *c)
		}
	})
}

func TestLockedWithLockUnlocksAfterPanic(t *testing.T) {
	var l logosync.Locked[int]
	func() {
		defer func() { _ = recover() }()
		l.WithLock(func(v *int) {
			*v = 1
			panic("boom")
		})
	}()
	done := make(chan struct{})
	go func() {
		defer close(done)
		l.WithLock(func(v *int) {
			*v = 2
		})
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("WithLock still locked after panic in f")
	}
	l.WithLock(func(v *int) {
		if *v != 2 {
			t.Fatalf("value = %d, want 2", *v)
		}
	})
}

func TestHotStoreLoad(t *testing.T) {
	h := new(logosync.Hot[lockConfig])
	if got := h.Load(); got != nil {
		t.Fatalf("zero Hot Load = %v, want nil", got)
	}
	v := &lockConfig{Port: 1}
	h.Store(v)
	if got := h.Load(); got != v {
		t.Fatalf("Load = %v, want stored pointer %v", got, v)
	}
}

func TestHotUpdateSequential(t *testing.T) {
	h := new(logosync.Hot[int])
	h.Update(func(old *int) *int {
		if old != nil {
			t.Errorf("old = %v, want nil", *old)
		}
		v := new(int)
		*v = 41
		return v
	})
	if got := h.Load(); got == nil || *got != 41 {
		t.Fatalf("Load after Update = %v, want 41", got)
	}
	h.Update(func(old *int) *int {
		v := new(int)
		*v = *old + 1
		return v
	})
	if got := h.Load(); got == nil || *got != 42 {
		t.Fatalf("Load after second Update = %v, want 42", got)
	}
}

type hotPoint struct {
	x, y int64
}

func TestHotUpdateConcurrent(t *testing.T) {
	const writers = 8
	const readers = 8
	const perWriter = 2000
	h := new(logosync.Hot[hotPoint])
	h.Store(&hotPoint{x: 1, y: 1})
	var wg sync.WaitGroup
	for w := range writers {
		wg.Go(func() {
			for i := range perWriter {
				n := int64(w*perWriter + i + 2)
				h.Update(func(old *hotPoint) *hotPoint {
					return &hotPoint{x: n, y: n}
				})
			}
		})
	}
	for range readers {
		wg.Go(func() {
			for range perWriter {
				p := h.Load()
				if p == nil {
					t.Error("Load returned nil after initial Store")
					return
				}
				if p.x != p.y {
					t.Errorf("torn value: x=%d y=%d", p.x, p.y)
					return
				}
			}
		})
	}
	wg.Wait()
	if p := h.Load(); p == nil || p.x != p.y {
		t.Fatalf("final value torn: %+v", p)
	}
}

func TestHotNoMemoryGrowth(t *testing.T) {
	const updates = 1000000
	h := new(logosync.Hot[hotPoint])
	h.Store(&hotPoint{x: 1, y: 1})
	runtime.GC()
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)
	start := time.Now()
	for i := range updates {
		n := int64(i + 2)
		h.Update(func(old *hotPoint) *hotPoint {
			return &hotPoint{x: n, y: n}
		})
	}
	elapsed := time.Since(start)
	runtime.GC()
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	t.Logf("1M updates in %v (%.0f updates/s)", elapsed, float64(updates)/elapsed.Seconds())
	if growth := int64(m1.HeapAlloc) - int64(m0.HeapAlloc); growth > 1<<20 {
		t.Fatalf("heap grew by %d bytes over 1M updates, want < 1MiB (retire-free)", growth)
	}
}

func TestHotLayout(t *testing.T) {
	var h logosync.Hot[hotPoint]
	if n := reflect.TypeOf(&h).Elem().NumField(); n != 1 {
		t.Fatalf("Hot has %d fields, want 1 (the atomic pointer; no retire list)", n)
	}
	var p atomic.Pointer[hotPoint]
	if unsafe.Sizeof(h) != unsafe.Sizeof(p) {
		t.Fatalf("Sizeof(Hot) = %d, want Sizeof(atomic.Pointer) = %d", unsafe.Sizeof(h), unsafe.Sizeof(p))
	}
	if unsafe.Alignof(h) != unsafe.Alignof(p) {
		t.Fatalf("Alignof(Hot) = %d, want Alignof(atomic.Pointer) = %d", unsafe.Alignof(h), unsafe.Alignof(p))
	}
}

func BenchmarkWithLock(b *testing.B) {
	var l logosync.Locked[int]
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		l.WithLock(func(v *int) { *v++ })
	}
}

func BenchmarkManualLock(b *testing.B) {
	var mu sync.Mutex
	var v int
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		func() {
			mu.Lock()
			defer mu.Unlock()
			v++
		}()
	}
}
