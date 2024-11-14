// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GOMAXPROCS=10 go test

// This is a copy of sync/rwmutex_test.go rewritten to test the
// runtime rwmutex.

package runtime_test

import (
	"fmt"
	. "runtime"
	"runtime/debug"
	"sync/atomic"
	"testing"
)

func parallelReader(m *RWMutex, clocked chan bool, cunlock *atomic.Bool, cdone chan bool) {
	m.RLock()
	clocked <- true
	for !cunlock.Load() {
	}
	m.RUnlock()
	cdone <- true
}

func doTestParallelReaders(numReaders int) {
	GOMAXPROCS(numReaders + 1)
	var m RWMutex
	m.Init()
	clocked := make(chan bool, numReaders)
	var cunlock atomic.Bool
	cdone := make(chan bool)
	for i := 0; i < numReaders; i++ {
		go parallelReader(&m, clocked, &cunlock, cdone)
	}
	// Wait for all parallel RLock()s to succeed.
	for i := 0; i < numReaders; i++ {
		<-clocked
	}
	cunlock.Store(true)
	// Wait for the goroutines to finish.
	for i := 0; i < numReaders; i++ {
		<-cdone
	}
}

func TestParallelRWMutexReaders(t *testing.T) {
	if GOARCH == "wasm" {
		t.Skip("wasm has no threads yet")
	}
	defer GOMAXPROCS(GOMAXPROCS(-1))
	// If runtime triggers a forced GC during this test then it will deadlock,
	// since the goroutines can't be stopped/preempted.
	// Disable GC for this test (see issue #10958).
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	// SetGCPercent waits until the mark phase is over, but the runtime
	// also preempts at the start of the sweep phase, so make sure that's
	// done too.
	GC()

	doTestParallelReaders(1)
	doTestParallelReaders(3)
	doTestParallelReaders(4)
}

func reader(rwm *RWMutex, num_iterations int, activity *int32, cdone chan bool) {
	for i := 0; i < num_iterations; i++ {
		rwm.RLock()
		n := atomic.AddInt32(activity, 1)
		if n < 1 || n >= 10000 {
			panic(fmt.Sprintf("wlock(%d)\n", n))
		}
		for i := 0; i < 100; i++ {
		}
		atomic.AddInt32(activity, -1)
		rwm.RUnlock()
	}
	cdone <- true
}

func writer(rwm *RWMutex, num_iterations int, activity *int32, cdone chan bool) {
	for i := 0; i < num_iterations; i++ {
		rwm.Lock()
		n := atomic.AddInt32(activity, 10000)
		if n != 10000 {
			panic(fmt.Sprintf("wlock(%d)\n", n))
		}
		for i := 0; i < 100; i++ {
		}
		atomic.AddInt32(activity, -10000)
		rwm.Unlock()
	}
	cdone <- true
}

func HammerRWMutex(gomaxprocs, numReaders, num_iterations int) {
	GOMAXPROCS(gomaxprocs)
	// Number of active readers + 10000 * number of active writers.
	var activity int32
	var rwm RWMutex
	rwm.Init()
	cdone := make(chan bool)
	go writer(&rwm, num_iterations, &activity, cdone)
	var i int
	for i = 0; i < numReaders/2; i++ {
		go reader(&rwm, num_iterations, &activity, cdone)
	}
	go writer(&rwm, num_iterations, &activity, cdone)
	for ; i < numReaders; i++ {
		go reader(&rwm, num_iterations, &activity, cdone)
	}
	// Wait for the 2 writers and all readers to finish.
	for i := 0; i < 2+numReaders; i++ {
		<-cdone
	}
}

func TestRWMutex(t *testing.T) {
	defer GOMAXPROCS(GOMAXPROCS(-1))
	n := 1000
	if testing.Short() {
		n = 5
	}
	HammerRWMutex(1, 1, n)
	HammerRWMutex(1, 3, n)
	HammerRWMutex(1, 10, n)
	HammerRWMutex(4, 1, n)
	HammerRWMutex(4, 3, n)
	HammerRWMutex(4, 10, n)
	HammerRWMutex(10, 1, n)
	HammerRWMutex(10, 3, n)
	HammerRWMutex(10, 10, n)
	HammerRWMutex(10, 5, n)
}

func BenchmarkRWMutexUncontended(b *testing.B) {
	type PaddedRWMutex struct {
		RWMutex
		pad [32]uint32
	}
	b.RunParallel(func { pb ->
		var rwm PaddedRWMutex
		rwm.Init()
		for pb.Next() {
			rwm.RLock()
			rwm.RLock()
			rwm.RUnlock()
			rwm.RUnlock()
			rwm.Lock()
			rwm.Unlock()
		}
	})
}

func benchmarkRWMutex(b *testing.B, localWork, writeRatio int) {
	var rwm RWMutex
	rwm.Init()
	b.RunParallel(func { pb ->
		foo := 0
		for pb.Next() {
			foo++
			if foo%writeRatio == 0 {
				rwm.Lock()
				rwm.Unlock()
			} else {
				rwm.RLock()
				for i := 0; i != localWork; i += 1 {
					foo *= 2
					foo /= 2
				}
				rwm.RUnlock()
			}
		}
		_ = foo
	})
}

func BenchmarkRWMutexWrite100(b *testing.B) {
	benchmarkRWMutex(b, 0, 100)
}

func BenchmarkRWMutexWrite10(b *testing.B) {
	benchmarkRWMutex(b, 0, 10)
}

func BenchmarkRWMutexWorkWrite100(b *testing.B) {
	benchmarkRWMutex(b, 100, 100)
}

func BenchmarkRWMutexWorkWrite10(b *testing.B) {
	benchmarkRWMutex(b, 100, 10)
}
