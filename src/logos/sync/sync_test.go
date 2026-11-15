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
	"runtime/debug"
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

func TestLockedWithLockHoldsLockDuringCallback(t *testing.T) {
	var l logosync.Locked[int]
	entered := make(chan struct{})
	release := make(chan struct{})
	firstDone := make(chan struct{})
	go func() {
		l.WithLock(func(v *int) {
			close(entered)
			<-release
			*v = 1
		})
		close(firstDone)
	}()
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("first WithLock callback did not start")
	}
	secondRan := make(chan int, 1)
	secondDone := make(chan struct{})
	go func() {
		l.WithLock(func(v *int) {
			secondRan <- *v
		})
		close(secondDone)
	}()
	select {
	case v := <-secondRan:
		t.Fatalf("second WithLock ran while the first callback holds the lock (saw %d)", v)
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	select {
	case <-secondDone:
	case <-time.After(10 * time.Second):
		t.Fatal("second WithLock did not run after release")
	}
	select {
	case <-firstDone:
	case <-time.After(10 * time.Second):
		t.Fatal("first WithLock did not return")
	}
	select {
	case v := <-secondRan:
		if v != 1 {
			t.Fatalf("second WithLock saw %d, want 1 (it must run after the first callback wrote)", v)
		}
	default:
		t.Fatal("second WithLock produced no observation")
	}
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
func TestHotStorePublishesUnderGC(t *testing.T) {
	h := new(logosync.Hot[hotPoint])
	h.Store(&hotPoint{x: 1, y: 1})
	stop := make(chan struct{})
	var gcLoop sync.WaitGroup
	gcLoop.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	})
	const writers = 4
	const perWriter = 2000
	var workers sync.WaitGroup
	for w := range writers {
		workers.Go(func() {
			for i := range perWriter {
				n := int64(w*perWriter + i + 2)
				h.Store(&hotPoint{x: n, y: n})
			}
		})
	}
	for range writers {
		workers.Go(func() {
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
	waitDone := make(chan struct{})
	go func() {
		workers.Wait()
		close(waitDone)
	}()
	select {
	case <-waitDone:
	case <-time.After(60 * time.Second):
		t.Fatal("workers did not finish")
	}
	close(stop)
	gcLoop.Wait()
	final := h.Load()
	if final == nil || final.x != final.y {
		t.Fatalf("final value torn or missing: %+v", final)
	}
	finalized := make(chan struct{}, 1)
	runtime.AddCleanup(final, func(ch chan struct{}) { ch <- struct{}{} }, finalized)
	h.Store(nil)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		runtime.GC()
		select {
		case <-finalized:
			return
		case <-deadline.C:
			t.Fatal("stored value was not finalized after Store(nil)")
		default:
		}
		runtime.Gosched()
	}
}

func TestHotKeepsValueReachable(t *testing.T) {
	type value struct {
		n    int
		keep *int
	}
	finalized := make(chan struct{}, 1)
	h := new(logosync.Hot[value])
	v := &value{n: 7, keep: new(int)}
	runtime.SetFinalizer(v, func(*value) { finalized <- struct{}{} })
	h.Store(v)
	v = nil
	for range 20 {
		runtime.GC()
		select {
		case <-finalized:
			t.Fatal("Hot value finalized while still stored")
		default:
		}
	}
	if got := h.Load(); got == nil || got.n != 7 {
		t.Fatalf("Hot value after GC = %v, want 7", got)
	}
	h.Store(nil)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		runtime.GC()
		select {
		case <-finalized:
			return
		case <-deadline.C:
			t.Fatal("Hot value was not finalized after Store(nil)")
		default:
		}
		runtime.Gosched()
	}
}

func waitHotFinalized(t *testing.T, finalized chan struct{}, what string) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		runtime.GC()
		select {
		case <-finalized:
			return
		case <-deadline.C:
			t.Fatalf("%s was not collectable: finalizer never ran after runtime.GC", what)
		default:
		}
		runtime.Gosched()
	}
}

func TestHotReplacedValueCollectable(t *testing.T) {
	finalized := make(chan struct{}, 2)
	h := new(logosync.Hot[hotPoint])
	v1 := &hotPoint{x: 1, y: 1}
	runtime.AddCleanup(v1, func(ch chan struct{}) { ch <- struct{}{} }, finalized)
	h.Store(v1)
	v1 = nil
	v2 := &hotPoint{x: 2, y: 2}
	runtime.AddCleanup(v2, func(ch chan struct{}) { ch <- struct{}{} }, finalized)
	h.Store(v2)
	v2 = nil
	waitHotFinalized(t, finalized, "value replaced by Store")
	h.Update(func(old *hotPoint) *hotPoint {
		return &hotPoint{x: old.x + 1, y: old.y + 1}
	})
	waitHotFinalized(t, finalized, "value replaced by Update")
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

func TestHotUpdateIncrementNoLostUpdates(t *testing.T) {
	const writers = 8
	const perWriter = 2000
	h := new(logosync.Hot[int])
	h.Store(new(int))
	var wg sync.WaitGroup
	for range writers {
		wg.Go(func() {
			for range perWriter {
				h.Update(func(old *int) *int {
					v := *old
					runtime.Gosched()
					n := new(int)
					*n = v + 1
					return n
				})
			}
		})
	}
	wg.Wait()
	if got := h.Load(); got == nil || *got != writers*perWriter {
		t.Fatalf("Load after %d increments = %v, want %d (a Load-then-Store Update loses updates)", writers*perWriter, got, writers*perWriter)
	}
}

func TestHotNoMemoryGrowth(t *testing.T) {
	const updates = 1000000
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
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
	if ft := reflect.TypeOf(&h).Elem().Field(0); ft.Type != reflect.TypeOf(atomic.Pointer[hotPoint]{}) {
		t.Fatalf("Hot field 0 type = %v, want %v (the value must live in an atomic.Pointer, not a bare word)", ft.Type, reflect.TypeOf(atomic.Pointer[hotPoint]{}))
	}
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
