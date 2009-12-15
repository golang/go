// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GOMAXPROCS=10 gotest

package sync_test

import (
	"runtime"
	. "sync"
	"testing"
)

func HammerSemaphore(s *uint32, loops int, cdone chan bool) {
	for i := 0; i < loops; i++ {
		runtime.Semacquire(s)
		runtime.Semrelease(s)
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
	runtime.GOMAXPROCS(2)
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

func BenchmarkUncontendedMutex(b *testing.B) {
	m := new(Mutex)
	HammerMutex(m, b.N, make(chan bool, 2))
}

func BenchmarkContendedMutex(b *testing.B) {
	b.StopTimer()
	m := new(Mutex)
	c := make(chan bool)
	runtime.GOMAXPROCS(2)
	b.StartTimer()

	go HammerMutex(m, b.N/2, c)
	go HammerMutex(m, b.N/2, c)
	<-c
	<-c
}
