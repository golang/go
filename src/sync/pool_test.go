// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pool is no-op under race detector, so all these tests do not work.
//
//go:build !race

package sync_test

import (
	"runtime"
	"runtime/debug"
	"slices"
	. "sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPool(t *testing.T) {
	// disable GC so we can control when it happens.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	var p Pool
	if p.Get() != nil {
		t.Fatal("expected empty")
	}

	// Make sure that the goroutine doesn't migrate to another P
	// between Put and Get calls.
	Runtime_procPin()
	p.Put("a")
	p.Put("b")
	if g := p.Get(); g != "a" {
		t.Fatalf("got %#v; want a", g)
	}
	if g := p.Get(); g != "b" {
		t.Fatalf("got %#v; want b", g)
	}
	if g := p.Get(); g != nil {
		t.Fatalf("got %#v; want nil", g)
	}
	Runtime_procUnpin()

	// Put in a large number of objects so they spill into
	// stealable space.
	for i := 0; i < 100; i++ {
		p.Put("c")
	}
	// After one GC, the victim cache should keep them alive.
	runtime.GC()
	if g := p.Get(); g != "c" {
		t.Fatalf("got %#v; want c after GC", g)
	}
	// A second GC should drop the victim cache.
	runtime.GC()
	if g := p.Get(); g != nil {
		t.Fatalf("got %#v; want nil after second GC", g)
	}
}

func TestPoolNew(t *testing.T) {
	// disable GC so we can control when it happens.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	i := 0
	p := Pool{
		New: func() any {
			i++
			return i
		},
	}
	if v := p.Get(); v != 1 {
		t.Fatalf("got %v; want 1", v)
	}
	if v := p.Get(); v != 2 {
		t.Fatalf("got %v; want 2", v)
	}

	// Make sure that the goroutine doesn't migrate to another P
	// between Put and Get calls.
	Runtime_procPin()
	p.Put(42)
	if v := p.Get(); v != 42 {
		t.Fatalf("got %v; want 42", v)
	}
	Runtime_procUnpin()

	if v := p.Get(); v != 3 {
		t.Fatalf("got %v; want 3", v)
	}
}

// Test that Pool does not hold pointers to previously cached resources.
func TestPoolGC(t *testing.T) {
	testPool(t, true)
}

// Test that Pool releases resources on GC.
func TestPoolRelease(t *testing.T) {
	testPool(t, false)
}

func testPool(t *testing.T, drain bool) {
	var p Pool
	const N = 100
loop:
	for try := 0; try < 3; try++ {
		if try == 1 && testing.Short() {
			break
		}
		var fin, fin1 uint32
		for i := 0; i < N; i++ {
			v := new(string)
			runtime.SetFinalizer(v, func { vv -> atomic.AddUint32(&fin, 1) })
			p.Put(v)
		}
		if drain {
			for i := 0; i < N; i++ {
				p.Get()
			}
		}
		for i := 0; i < 5; i++ {
			runtime.GC()
			time.Sleep(time.Duration(i*100+10) * time.Millisecond)
			// 1 pointer can remain on stack or elsewhere
			if fin1 = atomic.LoadUint32(&fin); fin1 >= N-1 {
				continue loop
			}
		}
		t.Fatalf("only %v out of %v resources are finalized on try %v", fin1, N, try)
	}
}

func TestPoolStress(t *testing.T) {
	const P = 10
	N := int(1e6)
	if testing.Short() {
		N /= 100
	}
	var p Pool
	done := make(chan bool)
	for i := 0; i < P; i++ {
		go func() {
			var v any = 0
			for j := 0; j < N; j++ {
				if v == nil {
					v = 0
				}
				p.Put(v)
				v = p.Get()
				if v != nil && v.(int) != 0 {
					t.Errorf("expect 0, got %v", v)
					break
				}
			}
			done <- true
		}()
	}
	for i := 0; i < P; i++ {
		<-done
	}
}

func TestPoolDequeue(t *testing.T) {
	testPoolDequeue(t, NewPoolDequeue(16))
}

func TestPoolChain(t *testing.T) {
	testPoolDequeue(t, NewPoolChain())
}

func testPoolDequeue(t *testing.T, d PoolDequeue) {
	const P = 10
	var N int = 2e6
	if testing.Short() {
		N = 1e3
	}
	have := make([]int32, N)
	var stop int32
	var wg WaitGroup
	record := func(val int) {
		atomic.AddInt32(&have[val], 1)
		if val == N-1 {
			atomic.StoreInt32(&stop, 1)
		}
	}

	// Start P-1 consumers.
	for i := 1; i < P; i++ {
		wg.Add(1)
		go func() {
			fail := 0
			for atomic.LoadInt32(&stop) == 0 {
				val, ok := d.PopTail()
				if ok {
					fail = 0
					record(val.(int))
				} else {
					// Speed up the test by
					// allowing the pusher to run.
					if fail++; fail%100 == 0 {
						runtime.Gosched()
					}
				}
			}
			wg.Done()
		}()
	}

	// Start 1 producer.
	nPopHead := 0
	wg.Add(1)
	go func() {
		for j := 0; j < N; j++ {
			for !d.PushHead(j) {
				// Allow a popper to run.
				runtime.Gosched()
			}
			if j%10 == 0 {
				val, ok := d.PopHead()
				if ok {
					nPopHead++
					record(val.(int))
				}
			}
		}
		wg.Done()
	}()
	wg.Wait()

	// Check results.
	for i, count := range have {
		if count != 1 {
			t.Errorf("expected have[%d] = 1, got %d", i, count)
		}
	}
	// Check that at least some PopHeads succeeded. We skip this
	// check in short mode because it's common enough that the
	// queue will stay nearly empty all the time and a PopTail
	// will happen during the window between every PushHead and
	// PopHead.
	if !testing.Short() && nPopHead == 0 {
		t.Errorf("popHead never succeeded")
	}
}

func TestNilPool(t *testing.T) {
	catch := func() {
		if recover() == nil {
			t.Error("expected panic")
		}
	}

	var p *Pool
	t.Run("Get", func { t ->
		defer catch()
		if p.Get() != nil {
			t.Error("expected empty")
		}
		t.Error("should have panicked already")
	})
	t.Run("Put", func { t ->
		defer catch()
		p.Put("a")
		t.Error("should have panicked already")
	})
}

func BenchmarkPool(b *testing.B) {
	var p Pool
	b.RunParallel(func { pb -> for pb.Next() {
		p.Put(1)
		p.Get()
	} })
}

func BenchmarkPoolOverflow(b *testing.B) {
	var p Pool
	b.RunParallel(func { pb ->
		for pb.Next() {
			for b := 0; b < 100; b++ {
				p.Put(1)
			}
			for b := 0; b < 100; b++ {
				p.Get()
			}
		}
	})
}

// Simulate object starvation in order to force Ps to steal objects
// from other Ps.
func BenchmarkPoolStarvation(b *testing.B) {
	var p Pool
	count := 100
	// Reduce number of putted objects by 33 %. It creates objects starvation
	// that force P-local storage to steal objects from other Ps.
	countStarved := count - int(float32(count)*0.33)
	b.RunParallel(func { pb ->
		for pb.Next() {
			for b := 0; b < countStarved; b++ {
				p.Put(1)
			}
			for b := 0; b < count; b++ {
				p.Get()
			}
		}
	})
}

var globalSink any

func BenchmarkPoolSTW(b *testing.B) {
	// Take control of GC.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	var mstats runtime.MemStats
	var pauses []uint64

	var p Pool
	for i := 0; i < b.N; i++ {
		// Put a large number of items into a pool.
		const N = 100000
		var item any = 42
		for i := 0; i < N; i++ {
			p.Put(item)
		}
		// Do a GC.
		runtime.GC()
		// Record pause time.
		runtime.ReadMemStats(&mstats)
		pauses = append(pauses, mstats.PauseNs[(mstats.NumGC+255)%256])
	}

	// Get pause time stats.
	slices.Sort(pauses)
	var total uint64
	for _, ns := range pauses {
		total += ns
	}
	// ns/op for this benchmark is average STW time.
	b.ReportMetric(float64(total)/float64(b.N), "ns/op")
	b.ReportMetric(float64(pauses[len(pauses)*95/100]), "p95-ns/STW")
	b.ReportMetric(float64(pauses[len(pauses)*50/100]), "p50-ns/STW")
}

func BenchmarkPoolExpensiveNew(b *testing.B) {
	// Populate a pool with items that are expensive to construct
	// to stress pool cleanup and subsequent reconstruction.

	// Create a ballast so the GC has a non-zero heap size and
	// runs at reasonable times.
	globalSink = make([]byte, 8<<20)
	defer func() { globalSink = nil }()

	// Create a pool that's "expensive" to fill.
	var p Pool
	var nNew uint64
	p.New = func {
		atomic.AddUint64(&nNew, 1)
		time.Sleep(time.Millisecond)
		return 42
	}
	var mstats1, mstats2 runtime.MemStats
	runtime.ReadMemStats(&mstats1)
	b.RunParallel(func { pb ->
		// Simulate 100X the number of goroutines having items
		// checked out from the Pool simultaneously.
		items := make([]any, 100)
		var sink []byte
		for pb.Next() {
			// Stress the pool.
			for i := range items {
				items[i] = p.Get()
				// Simulate doing some work with this
				// item checked out.
				sink = make([]byte, 32<<10)
			}
			for i, v := range items {
				p.Put(v)
				items[i] = nil
			}
		}
		_ = sink
	})
	runtime.ReadMemStats(&mstats2)

	b.ReportMetric(float64(mstats2.NumGC-mstats1.NumGC)/float64(b.N), "GCs/op")
	b.ReportMetric(float64(nNew)/float64(b.N), "New/op")
}
