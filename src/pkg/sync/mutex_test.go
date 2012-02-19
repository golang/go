// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GOMAXPROCS=10 go test

package sync_test

import (
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

func HammerSemaphore(s *uint32, loops int, cdone chan bool) {
	for i := 0; i < loops; i++ {
		Runtime_Semacquire(s)
		Runtime_Semrelease(s)
	}
	cdone <- true
}

func TestSemaphore(t *testing.T) {
	s := new(uint32)
	*s = 1
	c := make(chan bool)
	for i := 0; i < 10; i++ {
		go HammerSemaphore(s, 1000, c)
	}
	for i := 0; i < 10; i++ {
		<-c
	}
}

func BenchmarkUncontendedSemaphore(b *testing.B) {
	s := new(uint32)
	*s = 1
	HammerSemaphore(s, b.N, make(chan bool, 2))
}

func BenchmarkContendedSemaphore(b *testing.B) {
	b.StopTimer()
	s := new(uint32)
	*s = 1
	c := make(chan bool)
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	b.StartTimer()

	go HammerSemaphore(s, b.N/2, c)
	go HammerSemaphore(s, b.N/2, c)
	<-c
	<-c
}

func HammerMutex(m *Mutex, loops int, cdone chan bool) {
	for i := 0; i < loops; i++ {
		m.Lock()
		m.Unlock()
	}
	cdone <- true
}

func TestMutex(t *testing.T) {
	m := new(Mutex)
	c := make(chan bool)
	for i := 0; i < 10; i++ {
		go HammerMutex(m, 1000, c)
	}
	for i := 0; i < 10; i++ {
		<-c
	}
}

func TestMutexPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("unlock of unlocked mutex did not panic")
		}
	}()

	var mu Mutex
	mu.Lock()
	mu.Unlock()
	mu.Unlock()
}

func BenchmarkMutexUncontended(b *testing.B) {
	type PaddedMutex struct {
		Mutex
		pad [128]uint8
	}
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			var mu PaddedMutex
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					mu.Lock()
					mu.Unlock()
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func benchmarkMutex(b *testing.B, slack, work bool) {
	const (
		CallsPerSched  = 1000
		LocalWork      = 100
		GoroutineSlack = 10
	)
	procs := runtime.GOMAXPROCS(-1)
	if slack {
		procs *= GoroutineSlack
	}
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	var mu Mutex
	for p := 0; p < procs; p++ {
		go func() {
			foo := 0
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					mu.Lock()
					mu.Unlock()
					if work {
						for i := 0; i < LocalWork; i++ {
							foo *= 2
							foo /= 2
						}
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

func BenchmarkMutex(b *testing.B) {
	benchmarkMutex(b, false, false)
}

func BenchmarkMutexSlack(b *testing.B) {
	benchmarkMutex(b, true, false)
}

func BenchmarkMutexWork(b *testing.B) {
	benchmarkMutex(b, false, true)
}

func BenchmarkMutexWorkSlack(b *testing.B) {
	benchmarkMutex(b, true, true)
}
