// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	. "sync"
	"testing"
)

func BenchmarkSemaUncontended(b *testing.B) {
	type PaddedSem struct {
		sem uint32
		pad [32]uint32
	}
	b.RunParallel(func(pb *testing.PB) {
		sem := new(PaddedSem)
		for pb.Next() {
			Runtime_Semrelease(&sem.sem, false)
			Runtime_Semacquire(&sem.sem)
		}
	})
}

func benchmarkSema(b *testing.B, block, work bool) {
	if b.N == 0 {
		return
	}
	sem := uint32(0)
	if block {
		done := make(chan bool)
		go func() {
			for p := 0; p < runtime.GOMAXPROCS(0)/2; p++ {
				Runtime_Semacquire(&sem)
			}
			done <- true
		}()
		defer func() {
			<-done
		}()
	}
	b.RunParallel(func(pb *testing.PB) {
		foo := 0
		for pb.Next() {
			Runtime_Semrelease(&sem, false)
			if work {
				for i := 0; i < 100; i++ {
					foo *= 2
					foo /= 2
				}
			}
			Runtime_Semacquire(&sem)
		}
		_ = foo
		Runtime_Semrelease(&sem, false)
	})
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
