// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

func BenchmarkSemaUncontended(b *testing.B) {
	type PaddedSem struct {
		sem uint32
		pad [32]uint32
	}
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			sem := new(PaddedSem)
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					Runtime_Semrelease(&sem.sem)
					Runtime_Semacquire(&sem.sem)
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func benchmarkSema(b *testing.B, block, work bool) {
	const CallsPerSched = 1000
	const LocalWork = 100
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	c2 := make(chan bool, procs/2)
	sem := uint32(0)
	if block {
		for p := 0; p < procs/2; p++ {
			go func() {
				Runtime_Semacquire(&sem)
				c2 <- true
			}()
		}
	}
	for p := 0; p < procs; p++ {
		go func() {
			foo := 0
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					Runtime_Semrelease(&sem)
					if work {
						for i := 0; i < LocalWork; i++ {
							foo *= 2
							foo /= 2
						}
					}
					Runtime_Semacquire(&sem)
				}
			}
			c <- foo == 42
			Runtime_Semrelease(&sem)
		}()
	}
	if block {
		for p := 0; p < procs/2; p++ {
			<-c2
		}
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkSemaSyntNonblock(b *testing.B) {
	benchmarkSema(b, false, false)
}

func BenchmarkSemaSyntBlock(b *testing.B) {
	benchmarkSema(b, true, false)
}

func BenchmarkSemaWorkNonblock(b *testing.B) {
	benchmarkSema(b, false, true)
}

func BenchmarkSemaWorkBlock(b *testing.B) {
	benchmarkSema(b, true, true)
}
