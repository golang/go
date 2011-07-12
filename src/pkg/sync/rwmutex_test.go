// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GOMAXPROCS=10 gotest

package sync_test

import (
	"fmt"
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

func parallelReader(m *RWMutex, clocked, cunlock, cdone chan bool) {
	m.RLock()
	clocked <- true
	<-cunlock
	m.RUnlock()
	cdone <- true
}

func doTestParallelReaders(numReaders, gomaxprocs int) {
	runtime.GOMAXPROCS(gomaxprocs)
	var m RWMutex
	clocked := make(chan bool)
	cunlock := make(chan bool)
	cdone := make(chan bool)
	for i := 0; i < numReaders; i++ {
		go parallelReader(&m, clocked, cunlock, cdone)
	}
	// Wait for all parallel RLock()s to succeed.
	for i := 0; i < numReaders; i++ {
		<-clocked
	}
	for i := 0; i < numReaders; i++ {
		cunlock <- true
	}
	// Wait for the goroutines to finish.
	for i := 0; i < numReaders; i++ {
		<-cdone
	}
}

func TestParallelReaders(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(-1))
	doTestParallelReaders(1, 4)
	doTestParallelReaders(3, 4)
	doTestParallelReaders(4, 2)
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
	runtime.GOMAXPROCS(gomaxprocs)
	// Number of active readers + 10000 * number of active writers.
	var activity int32
	var rwm RWMutex
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
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(-1))
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

func TestRLocker(t *testing.T) {
	var wl RWMutex
	var rl Locker
	wlocked := make(chan bool, 1)
	rlocked := make(chan bool, 1)
	rl = wl.RLocker()
	n := 10
	go func() {
		for i := 0; i < n; i++ {
			rl.Lock()
			rl.Lock()
			rlocked <- true
			wl.Lock()
			wlocked <- true
		}
	}()
	for i := 0; i < n; i++ {
		<-rlocked
		rl.Unlock()
		select {
		case <-wlocked:
			t.Fatal("RLocker() didn't read-lock it")
		default:
		}
		rl.Unlock()
		<-wlocked
		select {
		case <-rlocked:
			t.Fatal("RLocker() didn't respect the write lock")
		default:
		}
		wl.Unlock()
	}
}

func BenchmarkRWMutexUncontended(b *testing.B) {
	type PaddedRWMutex struct {
		RWMutex
		pad [32]uint32
	}
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			var rwm PaddedRWMutex
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					rwm.RLock()
					rwm.RLock()
					rwm.RUnlock()
					rwm.RUnlock()
					rwm.Lock()
					rwm.Unlock()
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func benchmarkRWMutex(b *testing.B, localWork, writeRatio int) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	var rwm RWMutex
	for p := 0; p < procs; p++ {
		go func() {
			foo := 0
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
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
			}
			c <- foo == 42
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
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
