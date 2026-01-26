// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestNoRaceFin(t *testing.T) {
	c := make(chan bool)
	go func() {
		x := new(string)
		runtime.SetFinalizer(x, func(x *string) {
			*x = "foo"
		})
		*x = "bar"
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}

var finVar struct {
	sync.Mutex
	cnt int
}

func TestNoRaceFinGlobal(t *testing.T) {
	c := make(chan bool)
	go func() {
		x := new(string)
		runtime.SetFinalizer(x, func(x *string) {
			finVar.Lock()
			finVar.cnt++
			finVar.Unlock()
		})
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
	finVar.Lock()
	finVar.cnt++
	finVar.Unlock()
}

func TestRaceFin(t *testing.T) {
	c := make(chan bool)
	y := 0
	_ = y
	go func() {
		x := new(string)
		runtime.SetFinalizer(x, func(x *string) {
			y = 42
		})
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
	y = 66
}

func TestNoRaceCleanup(t *testing.T) {
	c := make(chan bool)
	go func() {
		x := new(string)
		y := new(string)
		runtime.AddCleanup(x, func(y *string) {
			*y = "foo"
		}, y)
		*y = "bar"
		runtime.KeepAlive(x)
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}

func TestRaceBetweenCleanups(t *testing.T) {
	// Allocate struct with pointer to avoid hitting tinyalloc.
	// Otherwise we can't be sure when the allocation will
	// be freed.
	type T struct {
		v int
		p unsafe.Pointer
	}
	sharedVar := new(int)
	v0 := new(T)
	v1 := new(T)
	cleanup := func(x int) {
		*sharedVar = x
	}
	runtime.AddCleanup(v0, cleanup, 0)
	runtime.AddCleanup(v1, cleanup, 0)
	v0 = nil
	v1 = nil

	runtime.GC()
	time.Sleep(100 * time.Millisecond)
}
