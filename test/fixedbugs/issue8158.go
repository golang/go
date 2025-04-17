// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"time"
)

func main() {
	c := make(chan bool, 1)
	go f1(c)
	<-c
	time.Sleep(10 * time.Millisecond)
	go f2(c)
	<-c
}

func f1(done chan bool) {
	defer func() {
		recover()
		done <- true
		runtime.Goexit() // left stack-allocated Panic struct on gp->panic stack
	}()
	panic("p")
}

func f2(done chan bool) {
	defer func() {
		recover()
		done <- true
		runtime.Goexit()
	}()
	time.Sleep(10 * time.Millisecond) // overwrote Panic struct with Timer struct
	runtime.GC()                      // walked gp->panic list, found mangled Panic struct, crashed
	panic("p")
}
