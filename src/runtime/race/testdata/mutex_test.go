// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"sync"
	"testing"
	"time"
)

func TestNoRaceMutex(t *testing.T) {
	var mu sync.Mutex
	var x int16 = 0
	ch := make(chan bool, 2)
	go func() {
		mu.Lock()
		defer mu.Unlock()
		x = 1
		ch <- true
	}()
	go func() {
		mu.Lock()
		x = 2
		mu.Unlock()
		ch <- true
	}()
	<-ch
	<-ch
}

func TestRaceMutex(t *testing.T) {
	var mu sync.Mutex
	var x int16 = 0
	ch := make(chan bool, 2)
	go func() {
		x = 1
		mu.Lock()
		defer mu.Unlock()
		ch <- true
	}()
	go func() {
		x = 2
		mu.Lock()
		mu.Unlock()
		ch <- true
	}()
	<-ch
	<-ch
}

func TestRaceMutex2(t *testing.T) {
	var mu1 sync.Mutex
	var mu2 sync.Mutex
	var x int8 = 0
	ch := make(chan bool, 2)
	go func() {
		mu1.Lock()
		defer mu1.Unlock()
		x = 1
		ch <- true
	}()
	go func() {
		mu2.Lock()
		x = 2
		mu2.Unlock()
		ch <- true
	}()
	<-ch
	<-ch
}

func TestNoRaceMutexPureHappensBefore(t *testing.T) {
	var mu sync.Mutex
	var x int16 = 0
	ch := make(chan bool, 2)
	go func() {
		x = 1
		mu.Lock()
		mu.Unlock()
		ch <- true
	}()
	go func() {
		<-time.After(1e5)
		mu.Lock()
		mu.Unlock()
		x = 1
		ch <- true
	}()
	<-ch
	<-ch
}

func TestNoRaceMutexSemaphore(t *testing.T) {
	var mu sync.Mutex
	ch := make(chan bool, 2)
	x := 0
	mu.Lock()
	go func() {
		x = 1
		mu.Unlock()
		ch <- true
	}()
	go func() {
		mu.Lock()
		x = 2
		mu.Unlock()
		ch <- true
	}()
	<-ch
	<-ch
}

// from doc/go_mem.html
func TestNoRaceMutexExampleFromHtml(t *testing.T) {
	var l sync.Mutex
	a := ""

	l.Lock()
	go func() {
		a = "hello, world"
		l.Unlock()
	}()
	l.Lock()
	_ = a
}

func TestRaceMutexOverwrite(t *testing.T) {
	c := make(chan bool, 1)
	var mu sync.Mutex
	go func() {
		mu = sync.Mutex{}
		c <- true
	}()
	mu.Lock()
	<-c
}
