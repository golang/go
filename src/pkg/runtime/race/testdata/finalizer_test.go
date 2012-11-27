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

func TestNoRaceFin(t *testing.T) {
	c := make(chan bool)
	go func() {
		x := new(int)
		runtime.SetFinalizer(x, func(x *int) {
			*x = 42
		})
		*x = 66
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(1e8)
}

var finVar struct {
	sync.Mutex
	cnt int
}

func TestNoRaceFinGlobal(t *testing.T) {
	c := make(chan bool)
	go func() {
		x := new(int)
		runtime.SetFinalizer(x, func(x *int) {
			finVar.Lock()
			finVar.cnt++
			finVar.Unlock()
		})
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(1e8)
	finVar.Lock()
	finVar.cnt++
	finVar.Unlock()
}

func TestRaceFin(t *testing.T) {
	c := make(chan bool)
	y := 0
	go func() {
		x := new(int)
		runtime.SetFinalizer(x, func(x *int) {
			y = 42
		})
		c <- true
	}()
	<-c
	runtime.GC()
	time.Sleep(1e8)
	y = 66
}
