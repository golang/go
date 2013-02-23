// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

var stop = make(chan bool, 1)

func perpetuumMobile() {
	select {
	case <-stop:
	default:
		go perpetuumMobile()
	}
}

func TestStopTheWorldDeadlock(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping during short test")
	}
	maxprocs := runtime.GOMAXPROCS(3)
	compl := make(chan bool, 2)
	go func() {
		for i := 0; i != 1000; i += 1 {
			runtime.GC()
		}
		compl <- true
	}()
	go func() {
		for i := 0; i != 1000; i += 1 {
			runtime.GOMAXPROCS(3)
		}
		compl <- true
	}()
	go perpetuumMobile()
	<-compl
	<-compl
	stop <- true
	runtime.GOMAXPROCS(maxprocs)
}

func TestYieldProgress(t *testing.T) {
	testYieldProgress(t, false)
}

func TestYieldLockedProgress(t *testing.T) {
	testYieldProgress(t, true)
}

func testYieldProgress(t *testing.T, locked bool) {
	c := make(chan bool)
	cack := make(chan bool)
	go func() {
		if locked {
			runtime.LockOSThread()
		}
		for {
			select {
			case <-c:
				cack <- true
				return
			default:
				runtime.Gosched()
			}
		}
	}()
	time.Sleep(10 * time.Millisecond)
	c <- true
	<-cack
}

func TestYieldLocked(t *testing.T) {
	const N = 10
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		for i := 0; i < N; i++ {
			runtime.Gosched()
			time.Sleep(time.Millisecond)
		}
		c <- true
		// runtime.UnlockOSThread() is deliberately omitted
	}()
	<-c
}

func TestBlockLocked(t *testing.T) {
	const N = 10
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		for i := 0; i < N; i++ {
			c <- true
		}
		runtime.UnlockOSThread()
	}()
	for i := 0; i < N; i++ {
		<-c
	}
}

func stackGrowthRecursive(i int) {
	var pad [128]uint64
	if i != 0 && pad[0] == 0 {
		stackGrowthRecursive(i - 1)
	}
}

func TestSchedLocalQueue(t *testing.T) {
	runtime.TestSchedLocalQueue1()
}

func TestSchedLocalQueueSteal(t *testing.T) {
	runtime.TestSchedLocalQueueSteal1()
}

func benchmarkStackGrowth(b *testing.B, rec int) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					stackGrowthRecursive(rec)
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkStackGrowth(b *testing.B) {
	benchmarkStackGrowth(b, 10)
}

func BenchmarkStackGrowthDeep(b *testing.B) {
	benchmarkStackGrowth(b, 1024)
}

func BenchmarkSyscall(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					runtime.Entersyscall()
					runtime.Exitsyscall()
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkSyscallWork(b *testing.B) {
	const CallsPerSched = 1000
	const LocalWork = 100
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			foo := 42
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					runtime.Entersyscall()
					for i := 0; i < LocalWork; i++ {
						foo *= 2
						foo /= 2
					}
					runtime.Exitsyscall()
				}
			}
			c <- foo == 42
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkCreateGoroutines(b *testing.B) {
	benchmarkCreateGoroutines(b, 1)
}

func BenchmarkCreateGoroutinesParallel(b *testing.B) {
	benchmarkCreateGoroutines(b, runtime.GOMAXPROCS(-1))
}

func benchmarkCreateGoroutines(b *testing.B, procs int) {
	c := make(chan bool)
	var f func(n int)
	f = func(n int) {
		if n == 0 {
			c <- true
			return
		}
		go f(n - 1)
	}
	for i := 0; i < procs; i++ {
		go f(b.N / procs)
	}
	for i := 0; i < procs; i++ {
		<-c
	}
}
