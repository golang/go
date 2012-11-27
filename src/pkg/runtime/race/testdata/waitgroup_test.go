// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestNoRaceWaitGroup(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	n := 1
	for i := 0; i < n; i++ {
		wg.Add(1)
		j := i
		go func() {
			x = j
			wg.Done()
		}()
	}
	wg.Wait()
}

func TestRaceWaitGroup(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	n := 2
	for i := 0; i < n; i++ {
		wg.Add(1)
		j := i
		go func() {
			x = j
			wg.Done()
		}()
	}
	wg.Wait()
}

func TestNoRaceWaitGroup2(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		x = 1
		wg.Done()
	}()
	wg.Wait()
	x = 2
}

// incrementing counter in Add and locking wg's mutex
func TestRaceWaitGroupAsMutex(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	c := make(chan bool, 2)
	go func() {
		wg.Wait()
		time.Sleep(100 * time.Millisecond)
		wg.Add(+1)
		x = 1
		wg.Add(-1)
		c <- true
	}()
	go func() {
		wg.Wait()
		time.Sleep(100 * time.Millisecond)
		wg.Add(+1)
		x = 2
		wg.Add(-1)
		c <- true
	}()
	<-c
	<-c
}

// Incorrect usage: Add is too late.
func TestRaceWaitGroupWrongWait(t *testing.T) {
	c := make(chan bool, 2)
	var x int
	var wg sync.WaitGroup
	go func() {
		wg.Add(1)
		runtime.Gosched()
		x = 1
		wg.Done()
		c <- true
	}()
	go func() {
		wg.Add(1)
		runtime.Gosched()
		x = 2
		wg.Done()
		c <- true
	}()
	wg.Wait()
	<-c
	<-c
}

// A common WaitGroup misuse that can potentially be caught be the race detector.
// For this simple case we must emulate Add() as read on &wg and Wait() as write on &wg.
// However it will have false positives if there are several concurrent Wait() calls.
func TestRaceFailingWaitGroupWrongAdd(t *testing.T) {
	c := make(chan bool, 2)
	var wg sync.WaitGroup
	go func() {
		wg.Add(1)
		wg.Done()
		c <- true
	}()
	go func() {
		wg.Add(1)
		wg.Done()
		c <- true
	}()
	wg.Wait()
	<-c
	<-c
}

func TestNoRaceWaitGroupMultipleWait(t *testing.T) {
	c := make(chan bool, 2)
	var wg sync.WaitGroup
	go func() {
		wg.Wait()
		c <- true
	}()
	go func() {
		wg.Wait()
		c <- true
	}()
	wg.Wait()
	<-c
	<-c
}

func TestNoRaceWaitGroupMultipleWait2(t *testing.T) {
	c := make(chan bool, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		wg.Done()
		wg.Wait()
		c <- true
	}()
	go func() {
		wg.Done()
		wg.Wait()
		c <- true
	}()
	wg.Wait()
	<-c
	<-c
}

// Correct usage but still a race
func TestRaceWaitGroup2(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		x = 1
		wg.Done()
	}()
	go func() {
		x = 2
		wg.Done()
	}()
	wg.Wait()
}

func TestNoRaceWaitGroupPanicRecover(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	defer func() {
		err := recover()
		if err != "sync: negative WaitGroup counter" {
			t.Fatalf("Unexpected panic: %#v", err)
		}
		x = 2
	}()
	x = 1
	wg.Add(-1)
}

// TODO: this is actually a panic-synchronization test, not a
// WaitGroup test. Move it to another *_test file
// Is it possible to get a race by synchronization via panic?
func TestNoRaceWaitGroupPanicRecover2(t *testing.T) {
	var x int
	var wg sync.WaitGroup
	ch := make(chan bool, 1)
	var f func() = func() {
		x = 2
		ch <- true
	}
	go func() {
		defer func() {
			err := recover()
			if err != "sync: negative WaitGroup counter" {
			}
			go f()
		}()
		x = 1
		wg.Add(-1)
	}()

	<-ch
}

func TestNoRaceWaitGroupTransitive(t *testing.T) {
	x, y := 0, 0
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		x = 42
		wg.Done()
	}()
	go func() {
		time.Sleep(1e7)
		y = 42
		wg.Done()
	}()
	wg.Wait()
	_ = x
	_ = y
}
