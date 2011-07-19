// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync/atomic"
	"testing"
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
		t.Logf("skipping during short test")
		return
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

func stackGrowthRecursive(i int) {
	var pad [128]uint64
	if i != 0 && pad[0] == 0 {
		stackGrowthRecursive(i - 1)
	}
}

func BenchmarkStackGrowth(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					stackGrowthRecursive(10)
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
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
